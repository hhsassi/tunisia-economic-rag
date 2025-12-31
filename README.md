# Tunisia Financial Data RAG

RAG system for analyzing Tunisia's economic indicators using LangChain and Groq.

## Tech Stack

- **LangChain** - RAG framework
- **Groq API** (Llama 3.1) - LLM
- **ChromaDB** - Vector database
- **Streamlit** - Web interface
- **HuggingFace** - Embeddings

## Features

- Web scraping of financial data sources
- Semantic search with vector embeddings
- Automatic table generation
- Multi-format export (CSV, JSON, Excel)
- Interactive Q&A interface

## Setup

```bash
pip install -r requirements.txt

# Create secrets file
mkdir .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your Groq API key

streamlit run streamlit_app.py
```

## Data Sources

- Université de Sherbrooke - Economic indicators
- INS Tunisia - Official statistics

## Usage

Ask questions like:
- "Show me Tunisia's GDP growth by year"
- "Create a table of inflation data"
- "Compare government vs private consumption"

## License

Educational and research purposes.