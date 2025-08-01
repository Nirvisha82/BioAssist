# BioAssist - Biomedical Web-Search Enabled RAG Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/RAG-Powered-green.svg" alt="RAG">
  <img src="https://img.shields.io/badge/AI-Gemini%202.0-orange.svg" alt="Gemini AI">
  <img src="https://img.shields.io/badge/Vector%20DB-FAISS-purple.svg" alt="FAISS">
</p>

BioAssist is an intelligent biomedical AI assistant that combines local knowledge base retrieval with live web search to provide accurate, grounded responses to healthcare and medical questions. Built with a robust RAG pipeline, it processes multiple document formats and ensures response reliability through advanced hallucination detection.

## 🚀 Features

- **🧠 Hybrid RAG System**: Combines local document retrieval with live web search
- **📚 Multi-Format Support**: Processes PDF, DOCX, TXT, and CSV files
- **🔍 Web-Enhanced Retrieval**: DuckDuckGo integration for real-time information
- **⚠️ Hallucination Detection**: Advanced grounding verification system
- **🛡️ Medical Focus Guardrails**: Ensures responses stay within biomedical domain
- **💬 Conversation Memory**: Maintains context across chat sessions
- **📊 Performance Metrics**: Real-time tracking of retrieval and generation metrics
- **🎨 Interactive UI**: Clean Streamlit interface with conversation management

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.13+ |
| **UI Framework** | Streamlit |
| **Vector Database** | FAISS |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) |
| **LLM** | Google Gemini 2.0 Flash |
| **Web Search** | DuckDuckGo Search |
| **Document Processing** | LangChain, PyPDF, pandas |
| **Deployment** | Docker |

## 🔧 Installation & Setup

### Prerequisites

- Python 3.13+
- Docker (optional)
- Google Gemini API Key

### Method A: Docker Deployment (Recommended)

1. **Clone Repository**
```bash
git clone <repository-url>
cd bioassist
```

2. **Configure API Key**
Edit `config/config.yaml` and add your Gemini API key:
```yaml
llm:
  api_key: "your_gemini_api_key_here"
  model_name: "gemini-2.0-flash-exp"
```

3. **Build Docker Image**
```bash
docker build -f docker/Dockerfile -t bioassist:latest .
```

4. **Run Container**
```bash
docker run --rm -p 8501:8501 bioassist:latest
```

5. **Access Application**
Navigate to `http://localhost:8501`

### Method B: Local Development

1. **Clone Repository**
```bash
git clone <repository-url>
cd bioassist
```

2. **Create Virtual Environment**
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Configure API Key**
Edit `config/config.yaml` with your Gemini API key.

5. **Launch Application**
```bash
streamlit run main.py
```

### Method C: VS Code Dev Containers

1. **Install Prerequisites**
   - Docker Desktop
   - VS Code with Dev Containers extension

2. **Open in Container**
   - Open folder in VS Code
   - Choose "Reopen in Container"
   - Or use Cmd/Ctrl+Shift+P → "Reopen in Container"

3. **Launch Application**
```bash
streamlit run main.py
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   RAG Pipeline  │    │  Document       │
│                 │────▶│                 │────▶│  Processor      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Search    │    │   Query         │    │   FAISS         │
│   (DuckDuckGo)  │◀───│   Processing    │────▶│   Vector DB     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Hallucination │    │   Gemini 2.0    │    │   Response      │
│   Detection     │◀───│   Generation    │────▶│   Assembly      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔄 RAG Pipeline Workflow

### 1. Document Processing
```bash
# Build vector index (optional - pre-built index included)
python src/ingestion/build_vector_index.py
```

**Process Flow:**
- **Document Ingestion**: Scans `data_source/{pdf,docx,txt,csv}` directories
- **Preprocessing & Chunking**: Splits documents into 500-token chunks with 100-token overlap
- **Embedding Generation**: Encodes chunks using all-MiniLM-L6-v2 (384-dim vectors)
- **Vector Index Building**: Stores embeddings in FAISS index with persistence

### 2. Query Processing

**Local Retrieval:**
- Query embedding generation
- FAISS similarity search (top-5 chunks)
- Similarity threshold filtering

**Web Search Integration:**
- DuckDuckGo search execution (max 5 results)
- Page content extraction and parsing
- Real-time content chunking and embedding

**Context Assembly:**
- Merge local + web chunks
- Enforce max context length (4,000 chars)
- Deduplicate overlapping content

### 3. Response Generation

**LLM Processing:**
- Context + query prompt assembly
- Gemini 2.0 Flash generation
- Response post-processing

**Quality Assurance:**
- Hallucination detection analysis
- Grounding verification
- Content safety filtering

## 📊 Configuration

### Core Settings (`config/config.yaml`)

```yaml
# LLM Configuration
llm:
  api_key: "your_gemini_api_key"
  model_name: "gemini-2.0-flash-exp"

# Retrieval Settings
retrieval:
  top_k: 5
  similarity_threshold: 0.3
  max_context_length: 4000

# Embeddings Configuration
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 500
  chunk_overlap: 100
  batch_size: 32

# Vector Database
vector_db:
  type: "faiss"
  persist_directory: "./vector_db"
  collection_name: "biomedical_documents"

# Web Search
web_search:
  num_results: 5
  timeout: 5

# Document Processing
document_processing:
  data_source_dir: "./data_source"
  supported_formats: [".pdf", ".docx", ".txt", ".csv"]
```

## 🎯 Key Features Deep Dive

### Hybrid RAG System
- **Local Knowledge Base**: Pre-processed medical documents with instant retrieval
- **Live Web Search**: Real-time medical information from trusted sources
- **Adaptive Fallback**: Automatic web search when local similarity is low

### Multi-Format Document Support
- **PDF**: Medical papers, research documents
- **DOCX**: Clinical guidelines, protocols
- **TXT**: Plain text medical resources
- **CSV**: Medical datasets with universal structure analysis

### Advanced Safety Features
- **Domain Guardrails**: Restricts responses to biomedical topics
- **Hallucination Detection**: Sentence-level grounding verification
- **Source Attribution**: Clear tracking of information sources
- **Confidence Scoring**: Reliability metrics for each response

### Conversation Management
- **Session Persistence**: Chat history saved across sessions
- **Context Awareness**: Maintains conversation flow
- **Export Options**: Download conversations in multiple formats
- **Conversation Analytics**: Performance metrics tracking

## 🧪 Usage Examples

### Basic Medical Query
```
User: "What are the symptoms of diabetes?"
BioAssist: Provides comprehensive answer with sources from local medical documents and recent web sources.
```

### Research Question
```
User: "Latest treatments for Alzheimer's disease"
BioAssist: Combines local research papers with current web search results for up-to-date information.
```


## 📈 Performance Metrics

The system tracks comprehensive metrics for each interaction:

- **Retrieval Time**: Local document search duration
- **Generation Time**: LLM response generation time
- **Context Length**: Total characters in prompt
- **Response Quality**: Token count and grounding ratio
- **Hallucination Detection**: Ungrounded statement count
- **Source Distribution**: Local vs. web source breakdown


## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/medical-enhancement`)
3. Commit changes (`git commit -m 'Add medical feature'`)
4. Push to branch (`git push origin feature/medical-enhancement`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Google Generative AI for Gemini 2.0 Flash
- Hugging Face for SentenceTransformers
- FAISS team for vector similarity search
- Streamlit for the interactive interface
- LangChain for document processing utilities

---

👥 Authors

- **[Nirvisha Soni](https://github.com/Nirvisha82)**

<p align="center">
  🧬 Built for the medical community with ❤️
</p>

<p align="center">
  <em>Advancing healthcare through intelligent information retrieval</em>
</p>
