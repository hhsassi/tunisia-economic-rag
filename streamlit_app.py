__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import asyncio
import logging
import time
from typing import List, Dict, Any
from datetime import datetime
from main import OptimizedFinancialDataRAG

st.set_page_config(
    page_title="Tunisia Economic Intelligence",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
        padding: 1.5rem;
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
    .stButton>button {
        width: 100%;
    }
    .metrics-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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

def display_chat_message(content: str, is_user: bool = False):
    """Display a chat message with appropriate styling"""
    message_class = "user-message" if is_user else "assistant-message"
    icon = "👤" if is_user else "🤖"
    
    # Escape HTML characters in content
    content_escaped = content.replace("<", "&lt;").replace(">", "&gt;")
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <strong>{icon}</strong> {content_escaped}
    </div>
    """, unsafe_allow_html=True)

def display_source(source: Dict, index: int):
    """Display a source document"""
    content = source.get('content', '')
    metadata = source.get('metadata', {})
    
    content_preview = content[:200] + "..." if len(content) > 200 else content
    content_preview = content_preview.replace("<", "&lt;").replace(">", "&gt;")
    
    st.markdown(f"""
    <div class="source-box">
        <strong>Source {index}</strong><br>
        <small>Type: {metadata.get('data_type', 'N/A')} | 
        Content: {metadata.get('content_type', 'N/A')}</small><br>
        {content_preview}
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🦅 Tunisia Economic Intelligence</h1>
    </div>
    """, unsafe_allow_html=True)

    # Initialize RAG system
    if st.session_state.rag_system is None:
        st.session_state.rag_system = OptimizedFinancialDataRAG()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        if not st.session_state.system_ready:
            st.info("💡 Load economic data to start chatting")
            
            if st.button("🚀 Load Economic Data", use_container_width=True):
                try:
                    # Read URLs from data.txt
                    with open("data.txt", "r", encoding="utf-8") as f:
                        urls = [line.strip() for line in f if line.strip()]
                    
                    if not urls:
                        st.error("No URLs found in data.txt")
                        return
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Load data with real-time progress
                    documents = []
                    total_urls = len(urls)
                    
                    status_text.text(f"📥 Loading 0/{total_urls} sources (0%)")
                    
                    async def load_single_url(url, index):
                        """Load a single URL and return documents"""
                        try:
                            loader_class = __import__('langchain_community.document_loaders', fromlist=['WebBaseLoader']).WebBaseLoader
                            loader = loader_class(url)
                            docs = loader.load()
                            return docs
                        except Exception as e:
                            st.warning(f"⚠️ Skipped URL {index+1}: {str(e)[:50]}")
                            return []
                    
                    # Load URLs one by one with progress updates
                    import asyncio
                    
                    async def load_all_with_progress():
                        all_docs = []
                        for i, url in enumerate(urls):
                            # Load this URL
                            docs = await load_single_url(url, i)
                            
                            # Process documents
                            for doc in docs:
                                doc.metadata.update({
                                    'source_url': url,
                                    'scraped_at': datetime.now().isoformat(),
                                    'data_type': st.session_state.rag_system._classify_data_type(url)
                                })
                            
                            all_docs.extend(docs)
                            
                            # Update progress AFTER loading (more realistic)
                            progress = ((i + 1) / total_urls)
                            progress_bar.progress(progress)
                            status_text.text(f"📥 Loading {i+1}/{total_urls} sources ({int(progress*100)}%)")
                        
                        # Final progress
                        progress_bar.progress(1.0)
                        status_text.text(f"📥 Loaded {total_urls}/{total_urls} sources (100%)")
                        return all_docs
                    
                    # Run the async loading
                    documents = asyncio.run(load_all_with_progress())
                    
                    if not documents:
                        st.error("❌ No documents could be loaded")
                        return
                    
                    # Build vector database
                    status_text.text("🔨 Building vector database...")
                    st.session_state.rag_system.build_vector_database(documents)
                    
                    # Setup QA chain
                    status_text.text("🔧 Setting up QA chain...")
                    st.session_state.rag_system.setup_qa_chain()
                    
                    # Done!
                    status_text.text(f"✅ Ready! Loaded {len(documents)} documents")
                    
                    st.session_state.system_ready = True
                    time.sleep(1.5)
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
                    logging.error(f"Error in load_data: {str(e)}")
        else:
            st.success("🟢 System Ready")
            
            # System stats
            try:
                stats = st.session_state.rag_system.get_system_stats()
                if stats.get("status") == "ready":
                    st.metric("Documents Loaded", stats.get("total_documents", 0))
            except:
                pass
            
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("🔄 Reload Data", use_container_width=True):
                st.session_state.system_ready = False
                st.session_state.rag_system = OptimizedFinancialDataRAG()
                st.session_state.messages = []
                st.rerun()

    # Main chat interface
    if st.session_state.system_ready:
        st.subheader("💬 Chat")
        
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                display_chat_message(message["content"], is_user=True)
            else:
                display_chat_message(message["content"], is_user=False)
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander(f"📚 View Sources ({len(message['sources'])})"):
                        for i, source in enumerate(message["sources"], 1):
                            display_source(source, i)

        # Chat input
        if prompt := st.chat_input("Ask about Tunisia's economy..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message(prompt, is_user=True)
            
            # Get AI response
            with st.spinner("🔍 Analyzing economic data..."):
                try:
                    result = st.session_state.rag_system.query(prompt)
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("source_documents", [])
                    })
                    
                    # Display response
                    display_chat_message(result["answer"], is_user=False)
                    
                    # Display sources
                    if result.get("source_documents"):
                        with st.expander(f"📚 View Sources ({len(result['source_documents'])})"):
                            for i, source in enumerate(result["source_documents"], 1):
                                display_source(source, i)
                                
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
                    display_chat_message(error_msg, is_user=False)
                    logging.error(f"Query error: {str(e)}")
        
        # Example questions (always visible after loading)
        st.markdown("---")
        st.subheader("💡 Example Questions:")
        
        questions = [
            "How did the 2011 Revolution impact Tunisia's GDP growth?",
            "What were the economic effects of COVID-19 on Tunisia?",
            "What is Tunisia's average inflation rate?"
        ]
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, q in enumerate(questions):
            with cols[i]:
                st.code(q, language=None)

    else:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="metrics-container" style="text-align: center; padding: 2rem;">
                <h3>Welcome to Tunisia Economic Intelligence</h3>
                <p style="color: #666;">
                    AI-powered analysis of Tunisia's economic data with historical context
                </p>
                <p style="color: #999; font-size: 0.9rem; margin-top: 1rem;">
                    Click "Load Economic Data" in the sidebar to begin
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("💡 Example Questions:")
        
        questions = [
            "How did the 2011 Revolution impact Tunisia's GDP growth?",
            "What were the economic effects of COVID-19 on Tunisia?",
            "What is Tunisia's average inflation rate?"
        ]
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, q in enumerate(questions):
            with cols[i]:
                st.code(q, language=None)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🦅 Tunisia Economic Intelligence System - Powered by RAG
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()