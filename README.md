# Lightweight Embedding Search Engine

A fast, efficient semantic search engine built with sentence transformers and FAISS (with numpy fallback). This project provides both a REST API and a Streamlit web interface for searching through document collections using semantic similarity.

## Features

- 🔍 **Semantic Search**: Uses sentence transformers to find documents by meaning, not just keywords
- ⚡ **Fast Indexing**: FAISS-powered vector search with automatic numpy fallback
- 💾 **Smart Caching**: SQLite-based embedding cache to avoid recomputing embeddings
- 🚀 **REST API**: FastAPI-based API for easy integration
- 🎨 **Web Interface**: Streamlit app for interactive searching
- 📦 **Lightweight**: Minimal dependencies, works with CPU-only setups

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lightweight-embedding-search-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset (optional - uses 20 Newsgroups dataset):
```bash
python -m src.download_dataset
```

## Quick Start

### How to Start API

Start the FastAPI server:

```bash
uvicorn src.api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. The server automatically:
- Initializes the search engine on startup
- Loads documents from the configured folder
- Builds the search index (using cached embeddings when available)
- Exposes REST endpoints for searching

**API Endpoints:**
- `GET /health` - Check API status
- `POST /search` - Search for documents
  ```json
  {
    "query": "your search query",
    "top_k": 5,
    "expand": false
  }
  ```
- `GET /docs/count` - Get number of indexed documents
- `POST /rebuild` - Rebuild the search index

**Alternative: Streamlit Interface**

In a new terminal:
```bash
streamlit run src/streamlit_app.py
```

## How to Run Embedding Generation

Embedding generation happens automatically during index building. The process:

1. **Automatic during `build_index()`**: When you call `engine.build_index()`, the system:
   - Checks the cache for each document's embedding
   - Computes embeddings only for documents not in cache or with changed content
   - Uses batch processing for efficiency

2. **Manual embedding generation**:
```python
from src.search_engine import SearchEngine

engine = SearchEngine(docs_folder="data/docs")
engine.load_docs()
engine.build_index()  # Embeddings generated here
```

3. **Force recomputation**: To regenerate all embeddings (ignoring cache):
```python
engine.build_index(force_recompute=True)
```

4. **Direct embedding access**:
```python
from src.embedder import Embedder

embedder = Embedder()
embedding = embedder.embed_one("your text here")
# or batch processing
embeddings = embedder.embed(["text1", "text2", "text3"])
```

The embedding model used is `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors). It automatically uses GPU if available, otherwise falls back to CPU.

## How Caching Works

The system uses SQLite-based caching to avoid recomputing embeddings for unchanged documents.

**Cache Mechanism:**

1. **Hash-based validation**: Each document's content is hashed using SHA256. The cache stores:
   - `doc_id`: Document identifier (filename without extension)
   - `filename`: Original filename
   - `hash`: SHA256 hash of the document content
   - `embedding`: Pickled embedding vector (BLOB)
   - `updated_at`: Timestamp of last update

2. **Cache lookup process**:
   - When building the index, for each document:
     - Compute current content hash
     - Check if cached entry exists with matching hash
     - If hash matches → use cached embedding
     - If hash differs or missing → compute new embedding and update cache

3. **Cache location**: 
   - Default: `embeddings_cache.db` in the project root
   - Configurable via `EMB_CACHE_DB` environment variable

4. **Cache benefits**:
   - Skips expensive embedding computation for unchanged documents
   - Significantly speeds up index rebuilding
   - Persists across application restarts

**Example cache flow:**
```python
# First run: computes and caches
engine.build_index()  # All embeddings computed

# Second run: uses cache
engine.build_index()  # All embeddings loaded from cache

# After document change: only changed docs recomputed
# (document content hash changes, cache miss triggers recomputation)
```

## Folder Structure

```
.
├── src/
│   ├── __init__.py           # Package initialization
│   ├── api.py                 # FastAPI server with REST endpoints
│   ├── search_engine.py       # Core search engine logic (indexing, searching)
│   ├── embedder.py            # Sentence transformer wrapper for embeddings
│   ├── cache_manager.py       # SQLite-based embedding cache manager
│   ├── utils.py               # Text processing utilities (cleaning, tokenization)
│   ├── streamlit_app.py       # Streamlit web interface
│   └── download_dataset.py    # Dataset downloader (20 Newsgroups)
├── data/
│   └── docs/                  # Document collection (.txt files)
│       └── [document files]   # One .txt file per document
├── embeddings_cache.db        # SQLite cache database (auto-generated)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

**Key Components:**
- `src/search_engine.py`: Orchestrates document loading, embedding generation, and search
- `src/cache_manager.py`: Manages SQLite database for embedding persistence
- `src/embedder.py`: Wraps sentence-transformers model with batch processing
- `src/api.py`: FastAPI application with lifecycle management
- `data/docs/`: Contains all `.txt` files to be indexed

## Usage Examples

### Python API

```python
from src.search_engine import SearchEngine

# Initialize and build index
engine = SearchEngine(docs_folder="data/docs")
engine.load_docs()
engine.build_index()

# Search
results = engine.search("machine learning", top_k=5)
for result in results:
    print(f"{result['doc_id']}: {result['score']:.3f}")
    print(result['preview'])
```

### REST API

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence", "top_k": 3}'
```

## Configuration

- `DOCS_FOLDER`: Environment variable to set document folder (default: `data/docs`)
- `EMB_CACHE_DB`: Environment variable to set cache database path (default: `embeddings_cache.db`)

## Design Choices

**Architecture Decisions:**

1. **FAISS with Numpy Fallback**:
   - Primary: FAISS (Facebook AI Similarity Search) for fast vector similarity search
   - Fallback: Pure NumPy implementation when FAISS unavailable
   - Rationale: FAISS provides 10-100x speedup for large datasets, but numpy ensures CPU-only compatibility

2. **SQLite for Caching**:
   - Lightweight, file-based database (no separate server needed)
   - Pickled embeddings stored as BLOB
   - Hash-based content validation for cache invalidation
   - Rationale: Simple persistence without external dependencies

3. **Sentence Transformers Model**:
   - Model: `all-MiniLM-L6-v2` (384 dimensions)
   - Rationale: Good balance between quality and speed, small model size (~80MB)

4. **Batch Embedding Processing**:
   - Processes multiple documents in batches
   - Rationale: More efficient GPU/CPU utilization than single-document processing

5. **Cosine Similarity**:
   - L2-normalized embeddings with inner product
   - Rationale: Standard for semantic similarity, works well with transformer embeddings

6. **FastAPI for REST API**:
   - Async-capable, auto-generated OpenAPI docs
   - Lifecycle management for engine initialization
   - Rationale: Modern, fast, and developer-friendly

7. **Text Preprocessing**:
   - Lowercasing, HTML tag removal, whitespace normalization
   - NLTK-based tokenization with stopword removal
   - Rationale: Clean input improves embedding quality

8. **Modular Design**:
   - Separate components: Embedder, CacheManager, SearchEngine
   - Rationale: Easy to test, maintain, and extend

## How It Works

1. **Document Loading**: Reads all `.txt` files from the specified folder
2. **Embedding Generation**: Uses `sentence-transformers/all-MiniLM-L6-v2` to generate embeddings
3. **Caching**: Stores embeddings in SQLite to avoid recomputation
4. **Indexing**: Builds FAISS index (or uses numpy) for fast similarity search
5. **Search**: Converts query to embedding and finds most similar documents using cosine similarity

## Requirements

- Python 3.7+
- See `requirements.txt` for full dependency list

## License

MIT

