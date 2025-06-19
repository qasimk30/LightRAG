from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import asyncio
import shutil
import tempfile
from datetime import datetime
import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from extensions.lightrag_extension import LightRAG_EXTENSIONS
from lightrag.utils import (
    Tokenizer,
    TiktokenTokenizer,
    EmbeddingFunc,
    always_get_an_event_loop,
    compute_mdhash_id,
    convert_response_to_json,
    lazy_external_import,
    priority_limit_async_func_call,
    get_content_summary,
    clean_text,
    check_storage_env_vars,
    logger,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Private Document Ingest & Search Service",
    description="Containerized service for private document ingestion and vector search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
WORKING_DIR = os.getenv("WORKING_DIR", "./private_documents")
SCREENSHOTS_BASE_PATH = os.getenv("SCREENSHOTS_PATH", "./screenshots")
STORAGE_PATH = os.getenv("STORAGE_PATH", "./storage")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8001"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "100")) * 1024 * 1024  # Default 100MB

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_BASE_PATH, exist_ok=True)
os.makedirs(STORAGE_PATH, exist_ok=True)
os.makedirs("./temp", exist_ok=True)

# Initialize embedding model globally
embedding_model = None
rag_instance = None

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # 1. Initialize the GenAI Client with your Gemini API Key
    client = genai.Client(api_key=GEMINI_API_KEY)

    # 2. Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    if system_prompt:
        combined_prompt += f"{system_prompt}\n"

    for msg in history_messages:
        # Each msg is expected to be a dict: {"role": "...", "content": "..."}
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"

    # 3. Call the Gemini model
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[combined_prompt],
        config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1),
    )

    # 4. Return the response text
    return response.text


async def embedding_func(texts: list[str]) -> np.ndarray:
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG instance on startup"""
    global rag_instance
    try:
        # Import your LightRAG extensions
        from extensions.lightrag_extension import LightRAG_EXTENSIONS
        from lightrag.kg.shared_storage import initialize_pipeline_status
        
        rag_instance = LightRAG_EXTENSIONS(
            working_dir=WORKING_DIR,
            screenshot_path=SCREENSHOTS_BASE_PATH,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=embedding_func,
            ),
        )

        await rag_instance.initialize_storages()
        await initialize_pipeline_status()
        logger.info("Private ingestion service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG instance: {str(e)}")
        raise

# Response Models
class ChunkResult(BaseModel):
    text_chunk: str
    screenshot_path: Optional[str] = None
    file_path: str
    page_number: Optional[int] = None
    relevance_score: float = 0.0
    chunk_id: str
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    chunks: List[ChunkResult]
    query: str
    total_chunks: int
    search_mode: str
    timestamp: str

class IngestionResponse(BaseModel):
    status: str
    message: str
    doc_id: Optional[str] = None
    file_path: str
    timestamp: str
    file_size: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service_name: str
    version: str

class StatusResponse(BaseModel):
    status: str
    storage_path: str
    screenshots_path: str
    working_dir: str
    total_documents: int
    total_chunks: int
    timestamp: str

# Dependency for RAG instance
def get_rag_instance():
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return rag_instance

# API Endpoints

@app.post("/ingest/document", response_model=IngestionResponse)
async def ingest_document(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    rag: Any = Depends(get_rag_instance)
):
    """
    Ingest a document into the private vector store
    """
    try:
        # Validate file size
        if file.size and file.size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        original_filename = Path(file.filename).name

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Create full file path with original name
        tmp_file_path = os.path.join(temp_dir, original_filename)

        # Save the file
        content = await file.read()
        with open(tmp_file_path, "wb") as f:
            f.write(content)
        
        try:
            # Use existing ainsert method
            await rag.ainsert(
                file_paths=[tmp_file_path],
                ids=[doc_id] if doc_id else None,
            )
            
            # Move file to permanent storage
            permanent_path = os.path.join(STORAGE_PATH, file.filename)
            shutil.move(tmp_file_path, permanent_path)
            
            logger.info(f"Successfully ingested document: {file.filename}")
            
            return IngestionResponse(
                status="success",
                message=f"Document {file.filename} ingested successfully",
                doc_id=doc_id or file.filename,
                file_path=permanent_path,
                timestamp=datetime.now().isoformat(),
                file_size=len(content)
            )
            
        finally:
            # Clean up temporary file if it still exists
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/query", response_model=QueryResponse)
async def vector_search(
    q: str, 
    mode: str = "hybrid", 
    top_k: int = 10,
    include_screenshots: bool = True,
    rag: Any = Depends(get_rag_instance)
):
    """
    Perform vector search and return relevant chunks with screenshots
    """
    try:
        # Validate parameters
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if mode not in ["naive", "local", "global", "hybrid", "mix"]:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
        
        if top_k < 1 or top_k > 100:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
        
        # Configure query parameters
        from lightrag.base import QueryParam
        param = QueryParam(
            mode=mode,
            top_k=top_k,
            only_need_context=True
        )
        
        # Use LightRAG's aretrieve_context method directly
        context = await rag.aretrieve_context(q, param)
        print(f"Retrieved context: {context}")
        
        # Convert context to chunks
        chunks = []
        if context:
            if isinstance(context, str):
                # Parse string context into chunks
                chunk_parts = context.split('\n---\n')  # Adjust separator as needed
                for i, chunk_text in enumerate(chunk_parts):
                    if chunk_text.strip():
                        chunks.append({
                            'content': chunk_text.strip(),
                            'screenshot_path': None,
                            'file_path': '',
                            'page_number': None,
                            'score': 1.0,
                            'id': f'chunk_{i}',
                            'metadata': {}
                        })
            elif isinstance(context, list):
                # Handle list of chunks
                for i, item in enumerate(context):
                    if isinstance(item, dict):
                        chunks.append(item)
                    else:
                        chunks.append({
                            'content': str(item),
                            'screenshot_path': None,
                            'file_path': '',
                            'page_number': None,
                            'score': 1.0,
                            'id': f'chunk_{i}',
                            'metadata': {}
                        })
        
        print(f"Processed chunks: {len(chunks)}")
        
        # Convert to response format
        chunk_results = []
        for chunk_data in chunks:
            # Process screenshot path
            screenshot_path = None
            if include_screenshots and chunk_data.get('screenshot_path'):
                screenshot_path = chunk_data['screenshot_path']
                # Ensure screenshot is accessible via API
                if not screenshot_path.startswith('http'):
                    screenshot_path = f"/screenshots/{os.path.basename(screenshot_path)}"
            
            chunk_result = ChunkResult(
                text_chunk=chunk_data.get('content', ''),
                screenshot_path=screenshot_path,
                file_path=chunk_data.get('file_path', ''),
                page_number=chunk_data.get('page_number'),
                relevance_score=chunk_data.get('score', 0.0),
                chunk_id=chunk_data.get('id', ''),
                metadata=chunk_data.get('metadata', {})
            )
            chunk_results.append(chunk_result)
        
        logger.info(f"Vector search completed. Found {len(chunk_results)} relevant chunks for query: {q}")
        
        return QueryResponse(
            chunks=chunk_results,
            query=q,
            total_chunks=len(chunk_results),
            search_mode=mode,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service_name="Private Document Ingest & Search Service",
        version="1.0.0"
    )

@app.get("/status", response_model=StatusResponse)
async def get_status(rag: Any = Depends(get_rag_instance)):
    """Get detailed service status"""
    try:
        # Get document and chunk counts
        total_documents = 0
        total_chunks = 0
        
        try:
            # This would depend on your RAG implementation
            if hasattr(rag, 'get_document_count'):
                total_documents = await rag.get_document_count()
            if hasattr(rag, 'get_chunk_count'):
                total_chunks = await rag.get_chunk_count()
        except:
            pass  # Counts are optional
        
        return StatusResponse(
            status="running",
            storage_path=STORAGE_PATH,
            screenshots_path=SCREENSHOTS_BASE_PATH,
            working_dir=WORKING_DIR,
            total_documents=total_documents,
            total_chunks=total_chunks,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/config")
async def get_configuration():
    """Get service configuration (non-sensitive)"""
    return {
        "working_dir": WORKING_DIR,
        "screenshots_path": SCREENSHOTS_BASE_PATH,
        "storage_path": STORAGE_PATH,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "service_port": SERVICE_PORT,
        "embedding_model": "all-MiniLM-L6-v2",
        "supported_modes": ["naive", "local", "global", "hybrid", "mix"],
        "version": "1.0.0"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found", "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting Private Ingestion Service on port {SERVICE_PORT}")
    logger.info(f"Working directory: {WORKING_DIR}")
    logger.info(f"Screenshots path: {SCREENSHOTS_BASE_PATH}")
    logger.info(f"Storage path: {STORAGE_PATH}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVICE_PORT,
        log_level="info",
        access_log=True
    )