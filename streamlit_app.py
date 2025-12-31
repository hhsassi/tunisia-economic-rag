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
from langchain_community.document_loaders import WebBaseLoader

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
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

def display_chat_message(content: str, is_user: bool = False):
    message_class = "user-message" if is_user else "assistant-message"
    icon = "👤" if is_user else "🤖"
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <strong>{icon}</strong> {content}
    </div>
    """, unsafe_allow_html=True)

def display_source(source: Dict, index: int):
    content = source.get('content', '')
    metadata = source.get('metadata', {})
    st.markdown(f"""
    <div class="source-box">
        <strong>Source {index}</strong><br>
        <small>Type: {metadata.get('data_type', 'N/A')} | 
        Content: {metadata.get('content_type', 'N/A')}</small><br>
        {content[:200]}...
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🦅 Tunisia Economic Intelligence</h1>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.rag_system is None:
        st.session_state.rag_system = OptimizedFinancialDataRAG()

    with st.sidebar:
        st.header("⚙️ System")
        
        if not st.session_state.system_ready:
            st.info("💡 Load data to start chatting")
            
            if st.button("🚀 Load Economic Data"):
                try:
                    with open("data.txt", "r", encoding="utf-8") as f:
                        urls = [line.strip() for line in f if line.strip()]
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Loading data sources...")
                    
                    async def load_data_with_progress():
                        docs = []
                        total_urls = len(urls)
                        
                        for i, url in enumerate(urls):
                            progress = int((i / total_urls) * 100)
                            progress_bar.progress(progress / 100)
                            status_text.text(f"Loading: {progress}% ({i}/{total_urls} URLs)")
                            
                            try:
                                loader = WebBaseLoader(url)
                                url_docs = loader.load()
                                for doc in url_docs:
                                    doc.metadata.update({
                                        'source_url': url,
                                        'scraped_at': datetime.now().isoformat()
                                    })
                                docs.extend(url_docs)
                            except Exception as e:
                                st.warning(f"Skipped {url}: {str(e)[:50]}")
                            
                            await asyncio.sleep(0.1)
                        
                        progress_bar.progress(100)
                        status_text.text(f"Loading: 100% ({total_urls}/{total_urls} URLs)")
                        return docs
                    
                    documents = asyncio.run(load_data_with_progress())
                    
                    if documents:
                        status_text.text("Building vector database...")
                        st.session_state.rag_system.build_vector_database(documents)
                        
                        status_text.text("Setting up QA chain...")
                        st.session_state.rag_system.setup_qa_chain()
                        
                        st.session_state.system_ready = True
                        status_text.text(f"✅ Ready! Loaded {len(documents)} documents")
                        time.sleep(1.5)
                        progress_bar.empty()
                        st.rerun()
                    else:
                        st.error("No documents loaded")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.success("🟢 System Ready")
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()

    if st.session_state.system_ready:
        st.subheader("💬 Chat")
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                display_chat_message(message["content"], is_user=True)
            else:
                display_chat_message(message["content"], is_user=False)
                
                if "sources" in message and message["sources"]:
                    with st.expander(f"📚 Sources ({len(message['sources'])})"):
                        for i, source in enumerate(message["sources"], 1):
                            display_source(source, i)

        if prompt := st.chat_input("Ask about Tunisia's economy..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message(prompt, is_user=True)
            
            with st.spinner("Analyzing..."):
                result = st.session_state.rag_system.query(prompt)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["source_documents"]
                })
                
                display_chat_message(result["answer"], is_user=False)
                
                if result["source_documents"]:
                    with st.expander(f"📚 Sources ({len(result['source_documents'])})"):
                        for i, source in enumerate(result["source_documents"], 1):
                            display_source(source, i)
        
        st.markdown("---")
        st.subheader("💡 Example Questions:")
        
        questions = [
            "📊 How did the 2011 Revolution impact Tunisia's GDP growth?",
            "🦠 What were the economic effects of COVID-19 on Tunisia?",
            "💰 What is Tunisia's average inflation rate?"
        ]
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, q in enumerate(questions):
            with cols[i]:
                st.info(q)

    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <h3>Welcome to Tunisia Economic Intelligence</h3>
                <p style="color: #666;">
                    AI-powered analysis of Tunisia's economic data with historical context
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("💡 Example Questions:")
        
        questions = [
            "📊 How did the 2011 Revolution impact Tunisia's GDP growth?",
            "🦠 What were the economic effects of COVID-19 on Tunisia?",
            "💰 What is Tunisia's average inflation rate?"
        ]
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        for i, q in enumerate(questions):
            with cols[i]:
                st.info(q)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🦅 Tunisia Economic Intelligence System
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()