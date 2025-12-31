import asyncio
import logging
import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import re
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedFinancialDataRAG:
    def __init__(self,
                 groq_model: str = "llama-3.1-8b-instant",
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "financial_data",
                 groq_api_key: str = None):
        """
        Initialize the Enhanced Financial Data RAG system with table generation capabilities
        """
        self.groq_model = groq_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        if groq_api_key is None:
            try:
                self.groq_api_key = st.secrets["groq"]["api_key"]
            except:
                raise ValueError("Groq API key not found in Streamlit secrets")
        else:
            self.groq_api_key = groq_api_key

        # === HuggingFace Embeddings (Free) ===
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # === Groq LLM ===
        self.llm = ChatGroq(
            model=self.groq_model,
            api_key=self.groq_api_key,
            temperature=0.2,
            max_tokens=1500
        )

        # Text splitter for different content types
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            length_function=len,
        )
        
        # Table-specific splitter (larger chunks for tables)
        self.table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

        self.vectorstore = None
        self.qa_chain = None

    async def process_urls_optimized(self, urls: List[str]) -> List[Document]:
        """Enhanced URL processing with better table detection and extraction"""
        documents = []
        failed_urls = []
        processing_stats = {
            'tables_extracted': 0,
            'text_chunks_created': 0,
            'fallback_documents': 0
        }

        logger.info(f"Starting optimized processing of {len(urls)} URLs...")

        for i, url in enumerate(urls, 1):
            try:
                logger.info(f"Processing URL {i}/{len(urls)}: {url}")
                loader = WebBaseLoader(url)
                docs = loader.load()

                for doc in docs:
                    # Enhanced metadata
                    doc.metadata.update({
                        'source_url': url,
                        'scraped_at': datetime.now().isoformat(),
                        'data_type': self._classify_data_type(url),
                        'processing_method': 'enhanced_extraction'
                    })

                    # Attempt to extract tables and structured data
                    extracted_docs = self._extract_structured_content(doc)
                    
                    if extracted_docs:
                        documents.extend(extracted_docs)
                        processing_stats['tables_extracted'] += len([d for d in extracted_docs if d.metadata.get('content_type') == 'table'])
                        processing_stats['text_chunks_created'] += len([d for d in extracted_docs if d.metadata.get('content_type') == 'text'])
                    else:
                        # Fallback to standard processing
                        doc.metadata['content_type'] = 'fallback_web'
                        documents.append(doc)
                        processing_stats['fallback_documents'] += 1

                logger.info(f"Successfully processed {url}")
                await asyncio.sleep(1)  # Respectful delay

            except Exception as e:
                logger.error(f"Failed to process {url}: {str(e)}")
                failed_urls.append(url)
                continue

        logger.info(f"Optimized processing completed. Stats: {processing_stats}")
        if failed_urls:
            logger.warning(f"Failed URLs: {failed_urls}")

        return documents

    def _extract_structured_content(self, document: Document) -> List[Document]:
        """Extract tables and structured content from documents"""
        extracted_docs = []
        content = document.page_content

        # Try to identify and extract tables
        table_patterns = [
            r'(\d{4})\s+([0-9.,]+(?:\s*%)?)',  # Year + Value patterns
            r'Year\s*[|\s]*\d{4}.*?(?=\n\n|\n[A-Z]|\Z)',  # Table headers with Year
            r'\|[^|]*\|[^|]*\|',  # Simple table format
            r'(?:GDP|Inflation|CPI|Population).*?\n(?:\d{4}.*?\n)+',  # Economic data tables
        ]

        # Extract potential tables
        tables_found = []
        for pattern in table_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            if matches:
                tables_found.extend(matches)

        # If we found structured data, create table documents
        if tables_found:
            # Create a table document
            table_content = self._format_extracted_table_data(content)
            if table_content:
                table_doc = Document(
                    page_content=table_content,
                    metadata={
                        **document.metadata,
                        'content_type': 'table',
                        'extraction_method': 'pattern_matching',
                        'table_rows': len(tables_found),
                        'original_length': len(content)
                    }
                )
                extracted_docs.append(table_doc)

        # Process remaining text content
        remaining_text = self._clean_text_content(content)
        if remaining_text and len(remaining_text.strip()) > 100:
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(remaining_text)
            
            for i, chunk in enumerate(text_chunks):
                if len(chunk.strip()) > 50:
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata={
                            **document.metadata,
                            'content_type': 'text',
                            'chunk_index': i,
                            'total_chunks': len(text_chunks)
                        }
                    )
                    extracted_docs.append(chunk_doc)

        return extracted_docs

    def _format_extracted_table_data(self, content: str) -> str:
        """Format extracted table data for better processing"""
        # Extract year-value pairs
        year_value_pattern = r'(\d{4})[^\d]*?([\d.,]+(?:\.\d+)?)\s*%?'
        matches = re.findall(year_value_pattern, content)
        
        if matches:
            formatted_data = "STRUCTURED FINANCIAL DATA:\n"
            formatted_data += "Year\tValue\tContext\n"
            
            for year, value in matches:
                # Try to find context around this data point
                context_pattern = rf'{year}[^0-9]*{re.escape(value)}[^0-9]*([^.]*)'
                context_match = re.search(context_pattern, content)
                context = context_match.group(1)[:100] if context_match else "Economic indicator"
                
                formatted_data += f"{year}\t{value}\t{context.strip()}\n"
            
            return formatted_data
        
        return None

    def _clean_text_content(self, text: str) -> str:
        """Clean and prepare text content"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove table artifacts
        text = re.sub(r'\|[\s\-]+\|', '', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()

    def _classify_data_type(self, url: str) -> str:
        """Enhanced data type classification"""
        classification_map = {
            'NY.GDP.MKTP.KD.ZG': 'GDP Growth Rate',
            'NE.CON.TOTL.CD': 'Total Consumption',
            'NE.CON.GOVT.CD': 'Government Consumption',
            'NE.CON.PRVT.CD': 'Private Consumption',
            'FP.CPI.TOTL': 'Consumer Price Index',
            'SI.DST.FRST': 'Income Distribution (First Quintile)',
            'SI.DST.02ND': 'Income Distribution (Second Quintile)',
            'SI.DST.03RD': 'Income Distribution (Third Quintile)',
            'SI.POV.DDAY': 'Poverty Rate',
            'NY.GDP.PCAP': 'GDP Per Capita',
            'PA.NUS': 'Exchange Rate',
            'NV.AGR.TOTL.ZS': 'Agriculture Value Added',
            'NV.IND': 'Industry Value Added',
        }
        
        for indicator, description in classification_map.items():
            if indicator in url:
                return description
        
        # Domain-based classification
        if 'ins.tn' in url:
            return 'National Institute of Statistics Data'
        elif 'imf.org' in url:
            return 'IMF Economic Data'
        elif 'worldbank.org' in url:
            return 'World Bank Data'
        else:
            return 'Economic Indicator'

    def build_vector_database(self, documents: List[Document]) -> None:
        """Build vector database with enhanced document processing"""
        logger.info(f"Building enhanced vector database with {len(documents)} documents...")
        
        try:
            # Process documents with enhanced cleaning
            processed_docs = []
            
            for doc in documents:
                # Enhanced content cleaning
                cleaned_content = self._enhance_document_content(doc.page_content)
                if len(cleaned_content.strip()) > 50:
                    doc.page_content = cleaned_content
                    processed_docs.append(doc)
            
            self.vectorstore = Chroma.from_documents(
                documents=processed_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            self.vectorstore.persist()
            
            logger.info(f"Enhanced vector database created with {len(processed_docs)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build vector database: {str(e)}")
            raise

    def _enhance_document_content(self, content: str) -> str:
        """Enhance document content for better retrieval"""
        # Add context markers for different types of content
        if "STRUCTURED FINANCIAL DATA:" in content:
            content = f"[TABLE_DATA] {content}"
        
        # Add economic indicators context
        economic_terms = ['GDP', 'inflation', 'CPI', 'consumption', 'poverty', 'exchange rate']
        for term in economic_terms:
            if term.lower() in content.lower():
                content = f"[{term.upper()}_DATA] {content}"
                break
        
        return content

    def load_vector_database(self) -> None:
        """Load existing vector database"""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            logger.info("Enhanced vector database loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vector database: {str(e)}")
            raise

    def setup_qa_chain(self) -> None:
        """Setup enhanced QA chain with better prompts for table generation"""
        if not self.vectorstore:
            raise ValueError("Vector database not initialized.")

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}  # Increased for more comprehensive results
        )

        # Enhanced prompt template for better table generation
        prompt_template = """
        You are an expert financial data analyst specializing in Tunisia's economic indicators. Your task is to provide comprehensive, structured responses that can be easily converted to tables and charts.

        When analyzing financial data, please:

        1. **Extract Specific Values**: Always include exact numbers, years, and percentages
        2. **Structure Your Response**: Organize data chronologically or by category
        3. **Identify Trends**: Point out increasing/decreasing patterns
        4. **Provide Context**: Explain what the numbers mean economically
        5. **Compare When Possible**: Reference regional or global benchmarks
        6. **Format for Tables**: Present data in a way that can be easily tabulated

        **Response Format Guidelines:**
        - Start with a clear summary
        - Include specific year-by-year data when available
        - Use consistent units and formatting
        - Highlight key insights and trends
        - End with economic implications

        Context from Tunisia's financial databases:
        {context}

        Question: {question}

        Provide a comprehensive, structured analysis with specific numerical data:
        """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        logger.info("Enhanced QA chain setup completed")

    def query(self, question: str) -> Dict[str, Any]:
        """Enhanced query method with source analysis"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please run setup_qa_chain() first.")

        try:
            logger.info(f"Processing enhanced query: {question}")
            result = self.qa_chain({"query": question})
            
            # Analyze sources for enhanced metadata
            sources_analysis = self._analyze_sources(result["source_documents"])
            
            return {
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ],
                "sources_analysis": sources_analysis,
                "timestamp": datetime.now().isoformat(),
                "query_type": self._classify_query_type(question)
            }
        except Exception as e:
            logger.error(f"Enhanced query failed: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error processing your query: {str(e)}",
                "source_documents": [],
                "sources_analysis": {},
                "timestamp": datetime.now().isoformat(),
                "query_type": "error"
            }

    def _analyze_sources(self, source_documents: List[Document]) -> Dict[str, int]:
        """Analyze source documents for metadata"""
        content_types = []
        data_types = []
        
        for doc in source_documents:
            content_types.append(doc.metadata.get('content_type', 'unknown'))
            data_types.append(doc.metadata.get('data_type', 'Unknown'))
        
        return {
            "content_types": dict(Counter(content_types)),
            "data_types": dict(Counter(data_types)),
            "total_sources": len(source_documents)
        }

    def _classify_query_type(self, question: str) -> str:
        """Classify the type of query for better response formatting"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['table', 'data', 'numbers', 'statistics']):
            return 'data_request'
        elif any(word in question_lower for word in ['trend', 'over time', 'years', 'evolution']):
            return 'trend_analysis'
        elif any(word in question_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            return 'comparative_analysis'
        elif any(word in question_lower for word in ['why', 'how', 'explain', 'reason']):
            return 'explanatory'
        else:
            return 'general_inquiry'

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        if not self.vectorstore:
            return {"status": "not_initialized"}
        
        try:
            # Get collection info
            collection = self.vectorstore.get()
            
            # Analyze document types
            content_types = []
            data_types = []
            
            for metadata in collection.get('metadatas', []):
                content_types.append(metadata.get('content_type', 'unknown'))
                data_types.append(metadata.get('data_type', 'Unknown'))
            
            return {
                "status": "ready",
                "total_documents": len(collection.get('documents', [])),
                "content_type_distribution": dict(Counter(content_types)),
                "data_type_distribution": dict(Counter(data_types)),
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def interactive_mode(self):
        """Enhanced interactive mode with better formatting"""
        print("\n" + "="*70)
        print("🦅 ENHANCED TUNISIA FINANCIAL DATA RAG SYSTEM")
        print("="*70)
        print("Features: ✨ Table Generation | 📊 Data Extraction | 📈 Trend Analysis")
        print("Ask questions about Tunisia's economic indicators!")
        print("Type 'exit', 'quit', or 'q' to stop")
        print("Type 'stats' to see system statistics")
        print("-"*70)

        while True:
            try:
                question = input("\n💬 Your question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye! Happy analyzing!")
                    break
                
                if question.lower() == 'stats':
                    stats = self.get_system_stats()
                    print(f"\n📊 System Statistics:")
                    print("-" * 30)
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                    continue
                
                if not question:
                    continue

                print("\n🔍 Analyzing with enhanced processing...")
                result = self.query(question)
                
                print(f"\n📊 Enhanced Analysis:")
                print("=" * 50)
                print(result["answer"])
                
                if result.get("sources_analysis"):
                    analysis = result["sources_analysis"]
                    print(f"\n📚 Sources Analysis:")
                    print(f"Total sources: {analysis.get('total_sources', 0)}")
                    
                    if analysis.get('content_types'):
                        print("Content types:", analysis['content_types'])
                
                print(f"\n🕒 Generated at: {result['timestamp']}")
                print(f"🔍 Query type: {result.get('query_type', 'unknown')}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye! Happy analyzing!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

async def main():
    """Enhanced main function with better error handling"""
    # Read URLs from data.txt
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("❌ File 'data.txt' not found. Please create it with your URLs.")
        return

    if not urls:
        print("❌ The file 'data.txt' is empty or contains no valid URLs.")
        return

    # Initialize enhanced RAG system
    rag = OptimizedFinancialDataRAG(
        groq_api_key="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"
    )

    print("=== Enhanced Financial Data RAG System ===")
    print(f"🚀 Processing {len(urls)} URLs with optimized extraction...")

    # Process URLs with enhanced extraction
    documents = await rag.process_urls_optimized(urls)

    if not documents:
        print("❌ No documents were retrieved. Please check your URLs and try again.")
        return

    print(f"📊 Processed {len(documents)} documents with enhanced extraction")
    
    # Analyze processing results
    content_types = [doc.metadata.get('content_type', 'unknown') for doc in documents]
    type_counts = Counter(content_types)
    
    print("📈 Processing Summary:")
    for content_type, count in type_counts.items():
        print(f"  - {content_type}: {count} documents")

    print("🏗️ Building enhanced vector database...")
    rag.build_vector_database(documents)

    print("⚙️ Setting up enhanced QA chain...")
    rag.setup_qa_chain()

    print("✅ Enhanced RAG system ready!")
    print("💡 Try asking for specific data tables, trends, or comparisons!")
    
    # Start enhanced interactive mode
    rag.interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())