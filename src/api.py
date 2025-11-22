import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .search_engine import SearchEngine

# ------------- Config -------------
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "data/docs")

# Global engine instance
engine = None

# ------------- Lifespan Events -------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global engine
    try:
        if not os.path.exists(DOCS_FOLDER):
            print(f"⚠️  Warning: Documents folder '{DOCS_FOLDER}' does not exist.")
            print(f"   Run 'python -m src.download_dataset' to download sample data.")
            engine = None
        else:
            print("🔹 Initializing search engine...")
            engine = SearchEngine(docs_folder=DOCS_FOLDER)
            print("📚 Loading documents...")
            doc_count = engine.load_docs()
            if doc_count == 0:
                print(f"⚠️  Warning: No documents found in '{DOCS_FOLDER}'")
                print(f"   Run 'python -m src.download_dataset' to download sample data.")
            else:
                print(f"   Found {doc_count} documents")
                print("🔨 Building index...")
                engine.build_index()
                print("✅ Search engine ready!")
    except Exception as e:
        print(f"❌ Error initializing search engine: {e}")
        import traceback
        traceback.print_exc()
        engine = None
    
    yield
    
    # Shutdown
    if engine and hasattr(engine, 'cache'):
        try:
            engine.cache.close()
        except:
            pass
    print("👋 Shutting down...")

# ------------- API App -------------
app = FastAPI(
    title="Lightweight Embedding Search Engine API",
    description="Semantic search using sentence transformers and FAISS",
    version="1.0.0",
    lifespan=lifespan
)

# ------------- Request Models -------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    expand: bool = False  # For future query expansion feature

# ------------- Routes -------------
@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Lightweight Embedding Search Engine API",
        "version": "1.0.0",
        "description": "Semantic search using sentence transformers and FAISS",
        "endpoints": {
            "health": "/health",
            "search": "/search (POST)",
            "docs_count": "/docs/count",
            "rebuild": "/rebuild (POST)",
            "api_docs": "/docs",
            "openapi_spec": "/openapi.json"
        },
        "status": "running" if engine is not None else "not initialized"
    }

@app.get("/health")
def health_check():
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    return {
        "status": "API running",
        "docs_loaded": len(engine.doc_texts) if engine.doc_texts else 0,
        "index_built": engine.index is not None or (engine.embeddings is not None)
    }

@app.post("/search")
def search(request: SearchRequest):
    """
    Search for documents similar to the query.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if request.top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")
    
    try:
        results = engine.search(request.query, top_k=request.top_k)
        return {
            "query": request.query,
            "top_k": request.top_k,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/docs/count")
def get_doc_count():
    """Get the number of documents in the index."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    return {
        "count": len(engine.doc_texts) if engine.doc_texts else 0,
        "indexed": len(engine.doc_ids) if engine.doc_ids else 0
    }

@app.post("/rebuild")
def rebuild_index(force_recompute: bool = False):
    """
    Rebuild the search index.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    try:
        engine.build_index(force_recompute=force_recompute)
        return {
            "status": "Index rebuilt successfully",
            "docs_indexed": len(engine.doc_ids) if engine.doc_ids else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild error: {str(e)}")
