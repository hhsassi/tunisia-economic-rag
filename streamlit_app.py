import os
import re
import json
import asyncio
import logging
import pandas as pd
import streamlit as st
from io import StringIO, BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional
from main import OptimizedFinancialDataRAG

st.set_page_config(
    page_title="Tunisia Financial Data RAG System - Enhanced",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    @staticmethod
    def extract_numerical_data(text: str) -> List[Dict[str, Any]]:
        data_points = []
        year_value_pattern = r'(\d{4})[^\d]*?([\d,.]+(?:\.\d+)?)\s*%?'
        matches = re.findall(year_value_pattern, text)
        
        for year, value in matches:
            try:
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
        
        all_text = rag_result['answer']
        for doc in rag_result.get('source_documents', []):
            all_text += " " + doc.get('content', '')
        
        numerical_data = TableResponseProcessor.extract_numerical_data(all_text)
        
        if numerical_data:
            df_main = pd.DataFrame(numerical_data)
            if not df_main.empty:
                df_main = df_main.sort_values('Year')
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
    files = {}
    if 'dataframe' in table_data and not table_data['dataframe'].empty:
        df = table_data['dataframe']
        
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        files['csv'] = {
            'content': csv_buffer.getvalue(),
            'filename': f"{table_title.replace(' ', '_').replace(':', '')}.csv",
            'mime': 'text/csv'
        }
        
        json_data = {
            'table_title': table_title,
            'generated_at': datetime.now().isoformat(),
            'data': df.to_dict('records'),
            'summary': {'total_rows': len(df), 'columns': list(df.columns)}
        }
        files['json'] = {
            'content': json.dumps(json_data, indent=2, default=str),
            'filename': f"{table_title.replace(' ', '_').replace(':', '')}.json",
            'mime': 'application/json'
        }
        
        try:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            files['excel'] = {
                'content': excel_buffer.getvalue(),
                'filename': f"{table_title.replace(' ', '_').replace(':', '')}.xlsx",
                'mime': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }
        except ImportError:
            pass
            
    return files


@st.cache_resource
def initialize_rag_system():
    try:
        rag = OptimizedFinancialDataRAG(
            groq_api_key="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"
        )
        return rag
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None


async def load_data_async(rag_system, urls):
    try:
        documents = await rag_system.process_urls_optimized(urls)
        if not documents:
            return False, "No documents were retrieved. Please check your URLs.", {}
        
        stats = {
            "total_documents": len(documents),
            "tables": len([d for d in documents if d.metadata.get('content_type') == 'table']),
            "text_chunks": len([d for d in documents if d.metadata.get('content_type') == 'text']),
            "fallback_docs": len([d for d in documents if d.metadata.get('content_type') == 'fallback_web']),
        }
        
        rag_system.build_vector_database(documents)
        rag_system.setup_qa_chain()
        return True, f"Successfully processed {len(documents)} documents", stats
    except Exception as e:
        return False, f"Error loading data: {str(e)}", {}


def load_urls_from_file():
    try:
        if os.path.exists("data.txt"):
            with open("data.txt", "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return []
    except Exception as e:
        st.error(f"Error reading data.txt: {str(e)}")
        return []


def display_enhanced_response(message_data: Dict[str, Any], message_index: int):
    st.markdown(f"""
    <div class="chat-message assistant-message">
        <strong>🤖 Assistant:</strong><br>
        {message_data['content']}
    </div>
    """, unsafe_allow_html=True)
    
    if 'enhanced_data' in message_data and message_data['enhanced_data']['tables']:
        st.markdown("### 📊 Generated Data Tables")
        
        for i, table_info in enumerate(message_data['enhanced_data']['tables']):
            with st.expander(f"📋 {table_info['title']}", expanded=(i == 0)):
                if 'dataframe' in table_info and not table_info['dataframe'].empty:
                    df = table_info['dataframe']
                    st.markdown('<div class="table-container">', unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True, height=min(400, len(df) * 35 + 100))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if table_info['type'] == 'time_series' and 'summary' in message_data['enhanced_data']:
                        summary = message_data['enhanced_data']['summary']
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Data Points", summary.get('total_data_points', 0))
                        col2.metric("Year Range", summary.get('year_range', 'N/A'))
                        col3.metric("Average Value", f"{summary.get('average_value', 0):.2f}")
                        col4.metric("Trend", summary.get('trend', 'N/A'))
                    
                    st.markdown('<div class="download-section">', unsafe_allow_html=True)
                    st.markdown("**📥 Download Options:**")
                    files = create_downloadable_files(table_info, table_info['title'])
                    
                    d_col1, d_col2, d_col3 = st.columns(3)
                    if 'csv' in files:
                        d_col1.download_button("📄 Download CSV", files['csv']['content'], files['csv']['filename'], files['csv']['mime'], key=f"csv_{message_index}_{i}")
                    if 'json' in files:
                        d_col2.download_button("📋 Download JSON", files['json']['content'], files['json']['filename'], files['json']['mime'], key=f"json_{message_index}_{i}")
                    if 'excel' in files:
                        d_col3.download_button("📊 Download Excel", files['excel']['content'], files['excel']['filename'], files['excel']['mime'], key=f"excel_{message_index}_{i}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No tabular data available for this response.")


def display_chat_message(message, is_user=True):
    message_class = "user-message" if is_user else "assistant-message"
    role = "👤 You" if is_user else "🤖 Assistant"
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <strong>{role}:</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)


def display_source_with_type(source, index):
    c_type = source['metadata'].get('content_type', 'unknown')
    d_type = source['metadata'].get('data_type', 'Unknown')
    p_method = source['metadata'].get('processing_method', 'unknown')
    
    if c_type == 'table':
        s_class, icon = "table-source", "📊"
        t_info = f"Table ({source['metadata'].get('rows', 'N/A')}×{source['metadata'].get('columns', 'N/A')})"
    else:
        s_class, icon = "text-source", "📝"
        t_info = "Text Content"
    
    st.markdown(f"""
    <div class="source-box {s_class}">
        <strong>{icon} Source {index}:</strong> 
        <span class="optimization-badge">{t_info}</span><br>
        <strong>Data Type:</strong> {d_type}<br>
        <strong>Processing:</strong> {p_method}<br>
        <small><strong>URL:</strong> {source['metadata'].get('source_url', 'Unknown')}</small><br>
        <hr style="margin: 0.5rem 0;">
        <em>{source['content']}</em>
    </div>
    """, unsafe_allow_html=True)


def main():
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

    with st.sidebar:
        st.header("⚙️ System Setup")
        if st.session_state.rag_system is None:
            with st.spinner("Initializing enhanced RAG system..."):
                st.session_state.rag_system = initialize_rag_system()
        
        if st.session_state.rag_system:
            st.success("✅ Enhanced RAG system initialized")
        else:
            st.error("❌ Failed to initialize RAG system")
            st.stop()
        
        st.info("🚀 **Features:**\n- Auto tables\n- CSV/JSON/Excel\n- Stats\n- Enhanced Extraction")
        st.subheader("📊 Data Loading")
        
        if st.button("📁 Load data from initial sources"):
            urls = load_urls_from_file()
            if urls:
                st.session_state.urls_to_load = urls
                st.success(f"Found {len(urls)} URLs")
            else:
                st.warning("No URLs found in data.txt")
        
        st.subheader("🔗 Manual URLs")
        manual_urls = st.text_area("Enter URLs (one per line):", height=100)
        
        if st.button("📥 Load Manual URLs"):
            if manual_urls.strip():
                urls = [u.strip() for u in manual_urls.split('\n') if u.strip()]
                st.session_state.urls_to_load = urls
                st.success(f"Added {len(urls)} URLs")
        
        if hasattr(st.session_state, 'urls_to_load') and not st.session_state.urls_loaded:
            if st.button("🚀 Build Database", type="primary"):
                with st.spinner("Processing..."):
                    success, msg, stats = asyncio.run(load_data_async(st.session_state.rag_system, st.session_state.urls_to_load))
                    if success:
                        st.session_state.system_ready = True
                        st.session_state.urls_loaded = True
                        st.session_state.processing_stats = stats
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        
        elif not st.session_state.system_ready and not st.session_state.urls_loaded:
            if st.button("📄 Load Existing Database"):
                try:
                    st.session_state.rag_system.load_vector_database()
                    st.session_state.rag_system.setup_qa_chain()
                    st.session_state.system_ready = True
                    st.success("✅ Database loaded")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.subheader("📈 Status")
        if st.session_state.system_ready:
            st.success("🟢 System Ready")
            if st.session_state.processing_stats:
                s = st.session_state.processing_stats
                st.write(f"📊 Tables: {s.get('tables', 0)}")
                st.write(f"📝 Chunks: {s.get('text_chunks', 0)}")
                st.write(f"📋 Total: {s.get('total_documents', 0)}")
        else:
            st.warning("🟡 Not Ready")
        
        if st.session_state.generated_tables:
            st.subheader("📥 Bulk Download")
            if st.button("📦 Export All Tables"):
                all_data = {
                    'generated_at': datetime.now().isoformat(),
                    'total_tables': len(st.session_state.generated_tables),
                    'tables': st.session_state.generated_tables
                }
                st.download_button("📋 Download JSON", json.dumps(all_data, indent=2, default=str), f"financial_tables_{datetime.now().strftime('%Y%m%d')}.json", "application/json")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.generated_tables = []
            st.rerun()

    if st.session_state.system_ready:
        st.subheader("💬 Enhanced Chat")
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                display_chat_message(msg["content"], True)
            else:
                if 'enhanced_data' in msg:
                    display_enhanced_response(msg, i)
                else:
                    display_chat_message(msg["content"], False)
                if "sources" in msg and msg["sources"]:
                    with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                        for j, src in enumerate(msg["sources"], 1):
                            display_source_with_type(src, j)

        if prompt := st.chat_input("Ask about Tunisia's economic data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message(prompt, True)
            
            with st.spinner("🔍 Analyzing..."):
                result = st.session_state.rag_system.query(prompt)
                enhanced = TableResponseProcessor.create_enhanced_response_with_tables(prompt, result)
                
                if enhanced['tables']:
                    for t in enhanced['tables']:
                        st.session_state.generated_tables.append({
                            'query': prompt,
                            'timestamp': datetime.now().isoformat(),
                            'title': t['title'],
                            'data': t['data']
                        })
                
                assistant_msg = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["source_documents"],
                    "enhanced_data": enhanced,
                    "timestamp": result["timestamp"]
                }
                st.session_state.messages.append(assistant_msg)
                display_enhanced_response(assistant_msg, len(st.session_state.messages) - 1)
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="metrics-container">
                <h3 style="text-align: center; color: #2d5aa0;">🚀 Enhanced RAG with Table Generation</h3>
                <p style="text-align: center;">Welcome to the Enhanced Tunisia Financial Data RAG System!</p>
                <ul>
                    <li>📊 <strong>Auto Tables:</strong> Structured responses</li>
                    <li>📥 <strong>Downloads:</strong> CSV, JSON, Excel support</li>
                    <li>📈 <strong>Stats:</strong> Auto-summary calculation</li>
                    <li>🎯 <strong>Extraction:</strong> Precision parsing</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("💡 Sample Questions:")
        q_list = [
            "📊 Show me Tunisia's GDP growth rates by year",
            "📈 Create a table of inflation data",
            "💰 Comparison table: government vs private consumption",
            "📉 Poverty rates over time table"
        ]
        c1, c2 = st.columns(2)
        for idx, q in enumerate(q_list):
            (c1 if idx % 2 == 0 else c2).info(q)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        🦅 Enhanced Tunisia Financial Data RAG System<br>
        <small>Generated Tables: {len(st.session_state.generated_tables)} | Ready for Export</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()