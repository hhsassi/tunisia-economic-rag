__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import pandas as pd
import json
import re
from io import StringIO
from main import OptimizedFinancialDataRAG  # Import your optimized RAG class

# Configure page
st.set_page_config(
    page_title="Tunisia Financial Data RAG System - Enhanced",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2d5aa0;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #ff6b6b;
    }
    .assistant-message {
        background-color: #e8f4f8;
        border-left-color: #2d5aa0;
    }
    .source-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .table-source {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
    }
    .text-source {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .metrics-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .optimization-badge {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .download-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .table-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False
if "urls_loaded" not in st.session_state:
    st.session_state.urls_loaded = False
if "processing_stats" not in st.session_state:
    st.session_state.processing_stats = {}
if "generated_tables" not in st.session_state:
    st.session_state.generated_tables = []

class TableResponseProcessor:
    """Enhanced processor for generating structured table responses"""
    
    @staticmethod
    def extract_numerical_data(text: str) -> List[Dict[str, Any]]:
        """Extract numerical data from text and structure it"""
        data_points = []
        
        # Pattern for year-value pairs
        year_value_pattern = r'(\d{4})[^\d]*?([\d,.]+(?:\.\d+)?)\s*%?'
        matches = re.findall(year_value_pattern, text)
        
        for year, value in matches:
            try:
                # Clean and convert value
                clean_value = re.sub(r'[^\d.-]', '', value)
                if clean_value:
                    data_points.append({
                        'Year': int(year),
                        'Value': float(clean_value),
                        'Original_Text': f"{year}: {value}"
                    })
            except (ValueError, TypeError):
                continue
        
        return data_points
    
    @staticmethod
    def create_enhanced_response_with_tables(query: str, rag_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced response with structured tables"""
        response_data = {
            'original_answer': rag_result['answer'],
            'tables': [],
            'summary': {},
            'metadata': {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'sources_count': len(rag_result.get('source_documents', []))
            }
        }
        
        # Extract data from the answer and sources
        all_text = rag_result['answer']
        for doc in rag_result.get('source_documents', []):
            all_text += " " + doc.get('content', '')
        
        # Extract numerical data
        numerical_data = TableResponseProcessor.extract_numerical_data(all_text)
        
        if numerical_data:
            # Create main data table
            df_main = pd.DataFrame(numerical_data)
            if not df_main.empty:
                # Sort by year
                df_main = df_main.sort_values('Year')
                
                # Calculate summary statistics
                response_data['summary'] = {
                    'total_data_points': len(df_main),
                    'year_range': f"{df_main['Year'].min()}-{df_main['Year'].max()}",
                    'value_range': f"{df_main['Value'].min():.2f} - {df_main['Value'].max():.2f}",
                    'average_value': df_main['Value'].mean(),
                    'trend': 'Increasing' if df_main['Value'].iloc[-1] > df_main['Value'].iloc[0] else 'Decreasing'
                }
                
                response_data['tables'].append({
                    'title': f'Financial Data for: {query}',
                    'data': df_main.to_dict('records'),
                    'dataframe': df_main,
                    'type': 'time_series'
                })
        
        # Create sources summary table
        sources_data = []
        for i, doc in enumerate(rag_result.get('source_documents', []), 1):
            sources_data.append({
                'Source_ID': f"Source_{i}",
                'Data_Type': doc.get('metadata', {}).get('data_type', 'Unknown'),
                'Content_Type': doc.get('metadata', {}).get('content_type', 'text'),
                'Processing_Method': doc.get('metadata', {}).get('processing_method', 'standard'),
                'URL': doc.get('metadata', {}).get('source_url', '')[:50] + '...' if doc.get('metadata', {}).get('source_url', '') else 'N/A'
            })
        
        if sources_data:
            df_sources = pd.DataFrame(sources_data)
            response_data['tables'].append({
                'title': 'Data Sources Summary',
                'data': df_sources.to_dict('records'),
                'dataframe': df_sources,
                'type': 'sources'
            })
        
        return response_data

def create_downloadable_files(table_data: Dict[str, Any], table_title: str) -> Dict[str, Any]:
    """Create downloadable files in multiple formats"""
    files = {}
    
    if 'dataframe' in table_data and not table_data['dataframe'].empty:
        df = table_data['dataframe']
        
        # CSV
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        files['csv'] = {
            'content': csv_buffer.getvalue(),
            'filename': f"{table_title.replace(' ', '_').replace(':', '')}.csv",
            'mime': 'text/csv'
        }
        
        # JSON
        json_data = {
            'table_title': table_title,
            'generated_at': datetime.now().isoformat(),
            'data': df.to_dict('records'),
            'summary': {
                'total_rows': len(df),
                'columns': list(df.columns)
            }
        }
        files['json'] = {
            'content': json.dumps(json_data, indent=2, default=str),
            'filename': f"{table_title.replace(' ', '_').replace(':', '')}.json",
            'mime': 'application/json'
        }
        
        # Excel (if needed)
        try:
            from io import BytesIO
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            files['excel'] = {
                'content': excel_buffer.getvalue(),
                'filename': f"{table_title.replace(' ', '_').replace(':', '')}.xlsx",
                'mime': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }
        except ImportError:
            pass  # openpyxl not available
    
    return files

@st.cache_resource
def initialize_rag_system():
    """Initialize the optimized RAG system (cached to avoid reinitialization)"""
    try:
        rag = OptimizedFinancialDataRAG(
            groq_api_key="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"
        )
        return rag
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

async def load_data_async(rag_system, urls):
    """Async function to load data with optimized processing"""
    try:
        # Process URLs with optimized extraction
        documents = await rag_system.process_urls_optimized(urls)
        
        if not documents:
            return False, "No documents were retrieved. Please check your URLs.", {}
        
        # Analyze processing statistics
        stats = {
            "total_documents": len(documents),
            "tables": len([d for d in documents if d.metadata.get('content_type') == 'table']),
            "text_chunks": len([d for d in documents if d.metadata.get('content_type') == 'text']),
            "fallback_docs": len([d for d in documents if d.metadata.get('content_type') == 'fallback_web']),
        }
        
        # Build vector database
        rag_system.build_vector_database(documents)
        
        # Setup QA chain
        rag_system.setup_qa_chain()
        
        return True, f"Successfully processed {len(documents)} documents", stats
    
    except Exception as e:
        return False, f"Error loading data: {str(e)}", {}

def load_urls_from_file():
    """Load URLs from data.txt file"""
    try:
        if os.path.exists("data.txt"):
            with open("data.txt", "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
            return urls
        else:
            return []
    except Exception as e:
        st.error(f"Error reading data.txt: {str(e)}")
        return []

def display_enhanced_response(message_data: Dict[str, Any], message_index: int):
    """Display enhanced response with tables and download options"""
    
    # Display main response
    st.markdown(f"""
    <div class="chat-message assistant-message">
        <strong>🤖 Assistant:</strong><br>
        {message_data['content']}
    </div>
    """, unsafe_allow_html=True)
    
    # Display tables if available
    if 'enhanced_data' in message_data and message_data['enhanced_data']['tables']:
        st.markdown("### 📊 Generated Data Tables")
        
        for i, table_info in enumerate(message_data['enhanced_data']['tables']):
            with st.expander(f"📋 {table_info['title']}", expanded=(i == 0)):
                
                # Display the table
                if 'dataframe' in table_info and not table_info['dataframe'].empty:
                    df = table_info['dataframe']
                    
                    # Table container
                    st.markdown('<div class="table-container">', unsafe_allow_html=True)
                    st.dataframe(
                        df, 
                        use_container_width=True,
                        height=min(400, len(df) * 35 + 100)
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Summary statistics for time series data
                    if table_info['type'] == 'time_series' and 'summary' in message_data['enhanced_data']:
                        summary = message_data['enhanced_data']['summary']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Data Points", summary.get('total_data_points', 0))
                        with col2:
                            st.metric("Year Range", summary.get('year_range', 'N/A'))
                        with col3:
                            st.metric("Average Value", f"{summary.get('average_value', 0):.2f}")
                        with col4:
                            st.metric("Trend", summary.get('trend', 'N/A'))
                    
                    # Download section
                    st.markdown('<div class="download-section">', unsafe_allow_html=True)
                    st.markdown("**📥 Download Options:**")
                    
                    # Create downloadable files
                    files = create_downloadable_files(table_info, table_info['title'])
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'csv' in files:
                            st.download_button(
                                label="📄 Download CSV",
                                data=files['csv']['content'],
                                file_name=files['csv']['filename'],
                                mime=files['csv']['mime'],
                                key=f"csv_{message_index}_{i}"
                            )
                    
                    with col2:
                        if 'json' in files:
                            st.download_button(
                                label="📋 Download JSON",
                                data=files['json']['content'],
                                file_name=files['json']['filename'],
                                mime=files['json']['mime'],
                                key=f"json_{message_index}_{i}"
                            )
                    
                    with col3:
                        if 'excel' in files:
                            st.download_button(
                                label="📊 Download Excel",
                                data=files['excel']['content'],
                                file_name=files['excel']['filename'],
                                mime=files['excel']['mime'],
                                key=f"excel_{message_index}_{i}"
                            )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No tabular data available for this response.")

def display_chat_message(message, is_user=True):
    """Display a chat message with proper styling"""
    message_class = "user-message" if is_user else "assistant-message"
    role = "👤 You" if is_user else "🤖 Assistant"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <strong>{role}:</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def display_source_with_type(source, index):
    """Display source with enhanced type information"""
    content_type = source['metadata'].get('content_type', 'unknown')
    data_type = source['metadata'].get('data_type', 'Unknown')
    processing_method = source['metadata'].get('processing_method', 'unknown')
    
    # Choose styling based on content type
    if content_type == 'table':
        source_class = "table-source"
        icon = "📊"
        type_info = f"Table ({source['metadata'].get('rows', 'N/A')}×{source['metadata'].get('columns', 'N/A')})"
    else:
        source_class = "text-source"
        icon = "📝"
        type_info = "Text Content"
    
    st.markdown(f"""
    <div class="source-box {source_class}">
        <strong>{icon} Source {index}:</strong> 
        <span class="optimization-badge">{type_info}</span><br>
        <strong>Data Type:</strong> {data_type}<br>
        <strong>Processing:</strong> {processing_method}<br>
        <small><strong>URL:</strong> {source['metadata'].get('source_url', 'Unknown')}</small><br>
        <hr style="margin: 0.5rem 0;">
        <em>{source['content']}</em>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🦅 Tunisia Financial Data RAG System - Enhanced</h1>
        <p style="color: white; text-align: center; margin: 0; font-size: 1.1rem;">
            <span class="optimization-badge">📊 Table Generation</span>
            <span class="optimization-badge">📥 Multi-Format Download</span>
            <span class="optimization-badge">⚡ Enhanced Processing</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for system setup
    with st.sidebar:
        st.header("⚙️ System Setup")
        
        # Initialize RAG system
        if st.session_state.rag_system is None:
            with st.spinner("Initializing enhanced RAG system..."):
                st.session_state.rag_system = initialize_rag_system()
        
        if st.session_state.rag_system is None:
            st.error("❌ Failed to initialize RAG system")
            st.stop()
        
        st.success("✅ Enhanced RAG system initialized")
        
        # Enhancement info
        st.info("""
        🚀 **Enhancement Features:**
        - 📊 Automatic table generation
        - 📥 CSV, JSON, Excel downloads
        - 📈 Statistical summaries
        - 🔍 Enhanced data extraction
        """)
        
        # Data loading section (same as before)
        st.subheader("📊 Data Loading")
        
        # Option 1: Load from data.txt
        if st.button("📁 Load data from initial sources"):
            urls = load_urls_from_file()
            if urls:
                st.session_state.urls_to_load = urls
                st.success(f"Found {len(urls)} URLs in data.txt")
            else:
                st.warning("No URLs found in data.txt or file doesn't exist")
        
        # Option 2: Manual URL input
        st.subheader("🔗 Or Add URLs Manually")
        manual_urls = st.text_area(
            "Enter URLs (one per line):",
            height=100,
            placeholder="https://example.com/economic-data\nhttps://another-site.com/gdp-data"
        )
        
        if st.button("📥 Load Manual URLs"):
            if manual_urls.strip():
                urls = [url.strip() for url in manual_urls.split('\n') if url.strip()]
                st.session_state.urls_to_load = urls
                st.success(f"Added {len(urls)} URLs")
            else:
                st.warning("Please enter at least one URL")
        
        # Load data button
        if hasattr(st.session_state, 'urls_to_load') and not st.session_state.urls_loaded:
            if st.button("🚀 Process URLs & Build Database", type="primary"):
                with st.spinner("Processing URLs with enhanced extraction..."):
                    # Run async function
                    success, message, stats = asyncio.run(
                        load_data_async(st.session_state.rag_system, st.session_state.urls_to_load)
                    )
                    
                    if success:
                        st.session_state.system_ready = True
                        st.session_state.urls_loaded = True
                        st.session_state.processing_stats = stats
                        st.success(message)
                        
                        # Display processing statistics
                        if stats:
                            st.markdown("### 📈 Processing Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Docs", stats.get("total_documents", 0))
                            with col2:
                                st.metric("Tables", stats.get("tables", 0))
                            with col3:
                                st.metric("Text Chunks", stats.get("text_chunks", 0))
                            with col4:
                                st.metric("Fallback", stats.get("fallback_docs", 0))
                        
                        st.rerun()
                    else:
                        st.error(message)
        
        # Try to load existing database
        elif not st.session_state.system_ready and not st.session_state.urls_loaded:
            if st.button("📄 Load Existing Database"):
                try:
                    st.session_state.rag_system.load_vector_database()
                    st.session_state.rag_system.setup_qa_chain()
                    st.session_state.system_ready = True
                    st.success("✅ Existing database loaded successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load existing database: {str(e)}")
        
        # System status and stats
        st.subheader("📈 System Status")
        if st.session_state.system_ready:
            st.success("🟢 System Ready")
            
            # Show processing stats if available
            if st.session_state.processing_stats:
                st.markdown("**Processing Statistics:**")
                stats = st.session_state.processing_stats
                st.write(f"📊 Tables: {stats.get('tables', 0)}")
                st.write(f"📝 Text Chunks: {stats.get('text_chunks', 0)}")
                st.write(f"🔄 Fallback Docs: {stats.get('fallback_docs', 0)}")
                st.write(f"📋 Total: {stats.get('total_documents', 0)}")
        else:
            st.warning("🟡 System Not Ready - Please load data first")
        
        # Download all tables button
        if st.session_state.generated_tables:
            st.subheader("📥 Bulk Download")
            if st.button("📦 Download All Generated Tables"):
                # Create a combined JSON with all tables
                all_tables_data = {
                    'generated_at': datetime.now().isoformat(),
                    'total_tables': len(st.session_state.generated_tables),
                    'tables': st.session_state.generated_tables
                }
                
                st.download_button(
                    label="📋 Download All Tables (JSON)",
                    data=json.dumps(all_tables_data, indent=2, default=str),
                    file_name=f"all_tunisia_financial_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.generated_tables = []
            st.rerun()

    # Main chat interface
    if st.session_state.system_ready:
        st.subheader("💬 Enhanced Chat with Table Generation")
        
        # Display chat messages with enhanced formatting
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                display_chat_message(message["content"], is_user=True)
            else:
                # Use enhanced display for assistant messages
                if 'enhanced_data' in message:
                    display_enhanced_response(message, i)
                else:
                    display_chat_message(message["content"], is_user=False)
                
                # Display sources
                if "sources" in message and message["sources"]:
                    with st.expander(f"📚 View Sources ({len(message['sources'])})"):
                        for j, source in enumerate(message["sources"], 1):
                            display_source_with_type(source, j)

        # Chat input with enhanced prompt
        if prompt := st.chat_input("Ask about Tunisia's economic data (optimized for table generation)..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message immediately
            display_chat_message(prompt, is_user=True)
            
            # Get response from RAG system
            with st.spinner("🔍 Generating enhanced response with tables..."):
                result = st.session_state.rag_system.query(prompt)
                
                # Process response for table generation
                enhanced_data = TableResponseProcessor.create_enhanced_response_with_tables(prompt, result)
                
                # Add to generated tables for bulk download
                if enhanced_data['tables']:
                    for table in enhanced_data['tables']:
                        st.session_state.generated_tables.append({
                            'query': prompt,
                            'timestamp': datetime.now().isoformat(),
                            'title': table['title'],
                            'data': table['data']
                        })
                
                # Add assistant message with enhanced data
                assistant_message = {
                    "role": "assistant", 
                    "content": result["answer"],
                    "sources": result["source_documents"],
                    "enhanced_data": enhanced_data,
                    "timestamp": result["timestamp"]
                }
                st.session_state.messages.append(assistant_message)
                
                # Display enhanced response
                display_enhanced_response(assistant_message, len(st.session_state.messages) - 1)

    else:
        # Enhanced welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="metrics-container">
                <h3 style="text-align: center; color: #2d5aa0;">🚀 Enhanced RAG with Table Generation</h3>
                <p style="text-align: center;">
                    Welcome to the Enhanced Tunisia Financial Data RAG System!<br><br>
                    <span class="optimization-badge">📊 Auto Table Generation</span>
                    <span class="optimization-badge">📥 Multi-Format Downloads</span><br><br>
                    New Features:
                </p>
                <ul>
                    <li>📊 <strong>Automatic Tables:</strong> Responses include structured data tables</li>
                    <li>📥 <strong>Downloads:</strong> CSV, JSON, Excel format support</li>
                    <li>📈 <strong>Statistics:</strong> Automatic summary calculations</li>
                    <li>🎯 <strong>Enhanced Extraction:</strong> Better numerical data parsing</li>
                    <li>📦 <strong>Bulk Export:</strong> Download all generated tables</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced sample questions
        st.subheader("💡 Sample Questions for Enhanced Table Generation:")
        
        table_questions = [
            "📊 Show me Tunisia's GDP growth rates by year in a table",
            "📈 Create a table of inflation data with trends",
            "💰 Generate a comparison table of government vs private consumption",
            "📉 Provide a structured table of poverty rates over time",
            "🏭 Show industrial sector contributions in tabular format",
            "💱 Create an exchange rate timeline table"
        ]
        
        col1, col2 = st.columns(2)
        
        for i, question in enumerate(table_questions):
            col = col1 if i % 2 == 0 else col2
            with col:
                st.info(question)

    # Enhanced footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🦅 Enhanced Tunisia Financial Data RAG System<br>
        <span class="optimization-badge">📊 Auto Table Generation</span>
        <span class="optimization-badge">📥 Multi-Format Download</span>
        <span class="optimization-badge">📈 Statistical Analysis</span><br>
        <small>Generated Tables: {len(st.session_state.generated_tables)} | Ready for Download</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()