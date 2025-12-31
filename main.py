import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import re
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedFinancialDataRAG:
    def __init__(self,
                 groq_model: str = "llama-3.1-8b-instant",
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "financial_data",
                 groq_api_key: str = None):
        
        self.groq_model = groq_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        if groq_api_key is None:
            try:
                self.groq_api_key = st.secrets["groq"]["api_key"]
            except:
                raise ValueError("Groq API key not found in secrets")
        else:
            self.groq_api_key = groq_api_key

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.llm = ChatGroq(
            model=self.groq_model,
            api_key=self.groq_api_key,
            temperature=0.2,
            max_tokens=2500
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
        )
        
        self.table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            length_function=len,
        )

        self.vectorstore = None
        self.qa_chain = None

    async def process_urls_optimized(self, urls: List[str]) -> List[Document]:
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
                    doc.metadata.update({
                        'source_url': url,
                        'scraped_at': datetime.now().isoformat(),
                        'data_type': self._classify_data_type(url),
                        'processing_method': 'enhanced_extraction'
                    })

                    extracted_docs = self._extract_structured_content(doc)
                    
                    if extracted_docs:
                        documents.extend(extracted_docs)
                        processing_stats['tables_extracted'] += len([d for d in extracted_docs if d.metadata.get('content_type') == 'table'])
                        processing_stats['text_chunks_created'] += len([d for d in extracted_docs if d.metadata.get('content_type') == 'text'])
                    else:
                        doc.metadata['content_type'] = 'fallback_web'
                        documents.append(doc)
                        processing_stats['fallback_documents'] += 1

                logger.info(f"Successfully processed {url}")
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Failed to process {url}: {str(e)}")
                failed_urls.append(url)
                continue

        logger.info(f"Optimized processing completed. Stats: {processing_stats}")
        if failed_urls:
            logger.warning(f"Failed URLs: {failed_urls}")

        return documents

    def _extract_structured_content(self, document: Document) -> List[Document]:
        extracted_docs = []
        content = document.page_content

        table_patterns = [
            r'(\d{4})\s+([0-9.,]+(?:\s*%)?)',
            r'Year\s*[|\s]*\d{4}.*?(?=\n\n|\n[A-Z]|\Z)',
            r'\|[^|]*\|[^|]*\|',
            r'(?:GDP|Inflation|CPI|Population).*?\n(?:\d{4}.*?\n)+',
        ]

        tables_found = []
        for pattern in table_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            if matches:
                tables_found.extend(matches)

        if tables_found:
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

        remaining_text = self._clean_text_content(content)
        if remaining_text and len(remaining_text.strip()) > 100:
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
        year_value_pattern = r'(\d{4})[^\d]*?([\d.,]+(?:\.\d+)?)\s*%?'
        matches = re.findall(year_value_pattern, content)
        
        if matches:
            formatted_data = "STRUCTURED FINANCIAL DATA:\n"
            formatted_data += "Year\tValue\tContext\n"
            
            for year, value in matches:
                context_pattern = rf'{year}[^0-9]*{re.escape(value)}[^0-9]*([^.]*)'
                context_match = re.search(context_pattern, content)
                context = context_match.group(1)[:100] if context_match else "Economic indicator"
                
                formatted_data += f"{year}\t{value}\t{context.strip()}\n"
            
            return formatted_data
        
        return ""

    def _clean_text_content(self, content: str) -> str:
        cleaned = re.sub(r'\n{3,}', '\n\n', content)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        return cleaned.strip()

    def _classify_data_type(self, url: str) -> str:
        url_lower = url.lower()
        
        if 'gdp' in url_lower or 'pib' in url_lower:
            return 'GDP'
        elif 'inflation' in url_lower or 'cpi' in url_lower:
            return 'Inflation'
        elif 'consumption' in url_lower or 'consommation' in url_lower:
            return 'Consumption'
        elif 'poverty' in url_lower or 'pauvrete' in url_lower:
            return 'Poverty'
        elif 'employment' in url_lower or 'emploi' in url_lower:
            return 'Employment'
        else:
            return 'Economic Data'

    def build_vector_database(self, documents: List[Document]):
        logger.info(f"Building vector database with {len(documents)} documents...")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        logger.info("Vector database built successfully")

    def load_vector_database(self):
        logger.info("Loading existing vector database...")
        
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        logger.info("Vector database loaded successfully")

    def setup_qa_chain(self):
        logger.info("Setting up QA chain...")
        
        prompt_template = """You are an economic analyst for Tunisia. Be CONCISE and ADAPTIVE.

RULES:
1. Length: Simple question → 2-3 sentences | Complex question → 1-2 paragraphs
2. Be direct: Start with the answer, use specific numbers/years
3. Context: Add historical events (2011 Revolution, COVID-19) ONLY if directly relevant
4. Vary structure: Don't repeat the same format for every answer
5. Never start with "Based on the context provided..."

Context: {context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("QA chain ready")

    def query(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            return {
                "answer": "System not initialized. Please load data first.",
                "source_documents": [],
                "timestamp": datetime.now().isoformat()
            }

        try:
            logger.info(f"Processing enhanced query: {question}")
            result = self.qa_chain({"query": question})
            
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
        if not self.vectorstore:
            return {"status": "not_initialized"}
        
        try:
            collection = self.vectorstore.get()
            
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
        print("\n" + "="*70)
        print("Tunisia Financial Data RAG System")
        print("="*70)
        print("Ask questions about Tunisia's economic indicators!")
        print("Type 'exit', 'quit', or 'q' to stop")
        print("Type 'stats' to see system statistics")
        print("-"*70)

        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                
                if question.lower() == 'stats':
                    stats = self.get_system_stats()
                    print(f"\nSystem Statistics:")
                    print("-" * 30)
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                    continue
                
                if not question:
                    continue

                print("\nAnalyzing...")
                result = self.query(question)
                
                print(f"\nAnswer:")
                print("=" * 50)
                print(result["answer"])
                
                if result.get("sources_analysis"):
                    analysis = result["sources_analysis"]
                    print(f"\nSources Analysis:")
                    print(f"Total sources: {analysis.get('total_sources', 0)}")
                
                print(f"\nGenerated at: {result['timestamp']}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

async def main():
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("File 'data.txt' not found. Please create it with your URLs.")
        return

    if not urls:
        print("The file 'data.txt' is empty or contains no valid URLs.")
        return

    rag = OptimizedFinancialDataRAG()

    print("=== Financial Data RAG System ===")
    print(f"Processing {len(urls)} URLs...")

    documents = await rag.process_urls_optimized(urls)

    if not documents:
        print("No documents were retrieved. Please check your URLs and try again.")
        return

    print(f"Processed {len(documents)} documents")
    
    content_types = [doc.metadata.get('content_type', 'unknown') for doc in documents]
    type_counts = Counter(content_types)
    
    print("Processing Summary:")
    for content_type, count in type_counts.items():
        print(f"  - {content_type}: {count} documents")

    print("Building vector database...")
    rag.build_vector_database(documents)

    print("Setting up QA chain...")
    rag.setup_qa_chain()

    print("RAG system ready!")
    
    rag.interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())