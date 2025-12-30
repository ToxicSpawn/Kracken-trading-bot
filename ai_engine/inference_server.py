"""
FastAPI inference server for fine-tuned LLMs.
Supports quantization, caching, rate limiting, and monitoring.
"""
import os
import hashlib
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import time

from .llm.fine_tuned_model import FineTunedLLM
from .rag.retrieval_system import RAGSystem
from .security.guardrails import SecurityGuard, PromptInjectionGuard
from .monitoring.tracker import get_tracker

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Inference Server",
    description="Fine-tuned LLM inference API with RAG support",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (lazy loaded)
_llm_model: Optional[FineTunedLLM] = None
_rag_system: Optional[RAGSystem] = None
_cache: Optional[Any] = None  # Redis client if available
_security_guard = SecurityGuard()
_injection_guard = PromptInjectionGuard()


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_length: int = Field(512, ge=1, le=4096, description="Maximum generation length")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(50, ge=1, description="Top-k sampling parameter")
    do_sample: bool = Field(True, description="Whether to use sampling")
    use_cache: bool = Field(True, description="Use Redis cache for responses")


class GenerateResponse(BaseModel):
    text: str
    cached: bool = False
    model: str


class RAGQueryRequest(BaseModel):
    query: str
    k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    use_hybrid: bool = True


class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]


# Rate limiting (simple in-memory, can be replaced with slowapi)
_rate_limit_store: Dict[str, List[datetime]] = defaultdict(list)
_rate_limit_window = timedelta(minutes=1)
_rate_limit_max_requests = 60


def check_rate_limit(client_id: str = "default") -> bool:
    """Simple rate limiting."""
    now = datetime.now()
    # Clean old entries
    _rate_limit_store[client_id] = [
        req_time for req_time in _rate_limit_store[client_id]
        if now - req_time < _rate_limit_window
    ]
    # Check limit
    if len(_rate_limit_store[client_id]) >= _rate_limit_max_requests:
        return False
    _rate_limit_store[client_id].append(now)
    return True


def get_llm_model() -> FineTunedLLM:
    """Lazy load LLM model."""
    global _llm_model
    if _llm_model is None:
        model_name = os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-v0.1")
        model_path = os.getenv("LLM_MODEL_PATH")
        _llm_model = FineTunedLLM(
            model_name=model_name,
            model_path=model_path,
            use_quantization=True,
        )
    return _llm_model


def get_rag_system() -> Optional[RAGSystem]:
    """Lazy load RAG system."""
    global _rag_system
    if _rag_system is None:
        rag_dir = os.getenv("RAG_PERSIST_DIR", str(Path(__file__).parent.parent / "rag_store"))
        try:
            _rag_system = RAGSystem(
                persist_directory=rag_dir,
                vectorstore_type="faiss",
            )
        except Exception as e:
            logger.warning(f"RAG system not available: {e}")
            return None
    return _rag_system


def get_cache():
    """Get Redis cache client if available."""
    global _cache
    if _cache is None:
        try:
            import redis
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            _cache = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
            _cache.ping()  # Test connection
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            _cache = False  # Mark as unavailable
    return _cache if _cache is not False else None


def get_cached_response(prompt: str, params: Dict[str, Any]) -> Optional[str]:
    """Get cached response if available."""
    cache = get_cache()
    if not cache:
        return None

    # Create cache key from prompt and params
    cache_key_data = f"{prompt}:{params}"
    cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
    cached = cache.get(cache_key)
    return cached


def set_cached_response(prompt: str, params: Dict[str, Any], response: str, ttl: int = 3600):
    """Cache response."""
    cache = get_cache()
    if not cache:
        return

    cache_key_data = f"{prompt}:{params}"
    cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
    cache.setex(cache_key, ttl, response)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": _llm_model is not None}


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        raise HTTPException(status_code=503, detail="Prometheus client not available")


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, client_id: str = "default"):
    """
    Generate text from a prompt.

    Supports caching and rate limiting.
    """
    # Rate limiting
    if not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Security validation
    validation = _security_guard.validate(request.prompt)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Security violation: {', '.join(validation['violations'])}"
        )

    if _injection_guard.detect(request.prompt):
        raise HTTPException(status_code=400, detail="Potential prompt injection detected")

    # Sanitize prompt
    sanitized_prompt = validation["sanitized"]

    # Check cache
    cached_response = None
    if request.use_cache:
        params = {
            "max_length": request.max_length,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "do_sample": request.do_sample,
        }
        cached_response = get_cached_response(sanitized_prompt, params)

    if cached_response:
        return GenerateResponse(
            text=cached_response,
            cached=True,
            model=os.getenv("LLM_MODEL_NAME", "unknown"),
        )

    # Generate
    start_time = time.time()
    try:
        llm = get_llm_model()
        generated_text = llm.generate(
            prompt=sanitized_prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
        )
        latency = time.time() - start_time

        # Log metrics
        tracker = get_tracker()
        tracker.log_inference(
            model_name=llm.model_name,
            latency=latency,
            tokens=len(generated_text.split()),
            status="success",
        )

        # Cache response
        if request.use_cache:
            params = {
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "do_sample": request.do_sample,
            }
            set_cached_response(sanitized_prompt, params, generated_text)

        return GenerateResponse(
            text=generated_text,
            cached=False,
            model=llm.model_name,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        tracker = get_tracker()
        tracker.log_inference(
            model_name=_llm_model.model_name if _llm_model else "unknown",
            latency=time.time() - start_time,
            tokens=0,
            status="error",
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """Query the RAG system."""
    rag = get_rag_system()
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not available")

    try:
        llm = get_llm_model()
        # Create a simple LLM wrapper for LangChain
        from langchain.llms.base import LLM as LangChainLLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        from typing import Any

        class LLMWrapper(LangChainLLM):
            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> str:
                return llm.generate(prompt, max_length=512, **kwargs)

            @property
            def _llm_type(self) -> str:
                return "fine_tuned_llm"

        langchain_llm = LLMWrapper()
        result = rag.query(
            query=request.query,
            llm=langchain_llm,
            k=request.k,
            use_hybrid=request.use_hybrid,
            return_source_documents=True,
        )

        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source", "unknown"),
                })

        return RAGQueryResponse(
            answer=result.get("result", ""),
            sources=sources,
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/add")
async def rag_add_documents(
    urls: Optional[List[str]] = None,
    file_paths: Optional[List[str]] = None,
):
    """Add documents to the RAG knowledge base."""
    rag = get_rag_system()
    if not rag:
        raise HTTPException(status_code=503, detail="RAG system not available")

    try:
        if urls:
            rag.load_from_urls(urls)
        if file_paths:
            for file_path in file_paths:
                rag.load_from_file(file_path)
        return {"status": "success", "added": len(urls or []) + len(file_paths or [])}
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio using Whisper.
    """
    try:
        import whisper
        model = whisper.load_model("base")

        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{file.filename}")
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Transcribe
        result = model.transcribe(str(temp_path))
        temp_path.unlink()  # Clean up

        return {"text": result["text"]}
    except ImportError:
        raise HTTPException(status_code=503, detail="Whisper not installed")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("INFERENCE_SERVER_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

