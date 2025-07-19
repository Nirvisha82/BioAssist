# BioAssist - Your Biomedical Web‑Enabled RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot pipeline tailored for biomedical data, with live web-search augmentation via DuckDuckGo. This project ingests local documents (PDF, DOCX, TXT, CSV), preprocesses, chunks and embed them using Huggingface's all-MiniLM-L6-v2, builds a vector index (FAISS), and provides scripts and a simple interface to query the index and retrieve contextual chunks.


## How to run: 

**Add GEMINI API Key** to `llm.api_key` in `config/config.yaml`

### A] Run directly using Docker.
- Build the docker image.
```bash
docker build -f docker/DockerFile -t rag-web-bot:latest .
```
- Run the created docker image
```bash
docker run --rm -p 8501:8501 rag-web-bot:latest
```
The streamlit app will run on:
`http://localhost:8501`
### B] Using CLI
#### 1 . Navigate to the folder

```http
  cd RAG-Chatbot
```
**NOTE:** Make sure `Python 3.13.3` is installed before proceeding.
#### 2. Create & activate a virtual environment
```bash
# Create the virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### C] Using VS Code Devcontainers
#### 1. Install Docker Desktop and VSCode Devcontainers extension
#### 2. Open the `RAG-Chatbot` folder in VScode.
#### 3. Opening the folder inside a container :
- Choose `Re-open in Container` when the option appears.
- Alternatively, hit `cmd+shift+p` or `ctrl+shift+p` and type `re-open in container`.
#### 4. A docker image will be created along with a container (first time)
#### 5. Once it shows that VS Code is connected to a Dev container, navigate to the integrated terminal.

### Launch the Streamlit UI

```bash
streamlit run main.py
```
### Document Processing Pipeline:
This step is not necessary to run the UI since a populated vector index is already provided. Incase a new one is to be build, delete the files in `vector_db` and run the following script:

```bash
python src/ingestion/build_vector_index.py 
#python3 if running on mac terminal
```

## Approach

### Data Processing:
**Document Ingestion** - Scan data_source/{pdf,docx,txt,csv} and load each file into memory. \\\
**Preprocessing & Chunking** - Split each document into 500-token chunks with 100-token overlap. \\\
**Embedding Generation** - Encode every chunk to a 384-dim vector using all-MiniLM-L6-v2 in batches of 32. \\\
**Vector Index Building** - Store all embeddings in a FAISS index (persisted to disk) for fast NN search. 

### RAG + Web Search:
**Local Retrieval** - Given a query, embed it and retrieve top-5 similar chunks from FAISS. \\\
**Live Web Search** - Always run a DuckDuckGo search (max 5 results), fetch & parse pages, chunk & embed live. \\\
**Context Assembly** - Merge local + web chunks, enforce max context length (4,000 chars), dedupe overlaps. \\\
**Answer Generation** - Feed merged context + question into Gemini-2.0-flash-lite to produce the final answer. \\\
**Safety & Guardrails** - Apply hallucination detection and content filters before returning the response. \\\
**Interface & Testing** - Provide CLI scripts (query_index.py, test_rag_web.py) and a Streamlit demo for end-to-end usage. 