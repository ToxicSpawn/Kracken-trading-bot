# LLM Upgrade Guide

This document describes the comprehensive LLM upgrades implemented to replace rule-based chatbots with fine-tuned LLMs and advanced AI features.

## Overview

The upgrade replaces the OpenAI-based `ai_client.py` with a complete fine-tuned LLM infrastructure including:

1. **Fine-tuned LLMs** with LoRA for efficient adaptation
2. **RAG (Retrieval-Augmented Generation)** for external knowledge
3. **FastAPI inference server** with quantization and caching
4. **Security guardrails** for input validation
5. **Monitoring** with W&B, MLflow, and Prometheus
6. **Web UI** with Gradio
7. **Voice support** with Whisper
8. **Docker deployment** with Kubernetes-ready configuration

## Architecture

### Core Components

#### 1. Fine-Tuned LLM (`ai_engine/llm/fine_tuned_model.py`)
- Supports Mistral-7B, Llama-3-8B, Phi-3 models
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- 4-bit quantization for faster inference
- Automatic device management (CUDA/CPU)

#### 2. RAG System (`ai_engine/rag/retrieval_system.py`)
- LangChain integration for document processing
- FAISS or ChromaDB vector storage
- Hybrid search (keyword + semantic)
- Support for URLs, files, and directories

#### 3. Inference Server (`ai_engine/inference_server.py`)
- FastAPI-based REST API
- Redis caching for frequent queries
- Rate limiting
- Prometheus metrics export
- Whisper transcription endpoint

#### 4. Security (`ai_engine/security/guardrails.py`)
- Input sanitization
- Prompt injection detection
- Content filtering (profanity, toxic language, PII)

#### 5. Monitoring (`ai_engine/monitoring/tracker.py`)
- Weights & Biases integration
- MLflow tracking
- Prometheus metrics

#### 6. UI (`ai_engine/ui/gradio_app.py`)
- Gradio web interface
- Text generation
- RAG query interface
- Real-time interaction

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export LLM_MODEL_NAME="mistralai/Mistral-7B-v0.1"  # or microsoft/Phi-3-mini-4k-instruct
export LLM_MODEL_PATH="./models/fine-tuned"  # Optional: path to fine-tuned adapter
export RAG_PERSIST_DIR="./rag_store"
export REDIS_HOST="localhost"
export REDIS_PORT=6379
```

### 3. Fine-Tune a Model (Optional)

```bash
python scripts/train_llm.py \
    --model-name mistralai/Mistral-7B-v0.1 \
    --data-path ./data/training_data.jsonl \
    --output-dir ./models/fine-tuned \
    --epochs 3 \
    --batch-size 4
```

### 4. Set Up RAG Knowledge Base

```bash
python scripts/setup_rag.py \
    --urls https://example.com/docs \
    --files ./docs/README.md \
    --directories ./docs \
    --persist-dir ./rag_store
```

### 5. Launch Inference Server

```bash
python -m ai_engine.inference_server
```

Server will be available at `http://localhost:8000`

### 6. Launch Web UI

```bash
python scripts/launch_ui.py --port 7860
```

UI will be available at `http://localhost:7860`

## Using Docker

### Build and Run

```bash
docker-compose up -d
```

This starts:
- Inference server (port 8000)
- Gradio UI (port 7860)
- Redis (port 6379)
- Prometheus (port 9090)
- Grafana (port 3000)

### Environment Variables

Create a `.env` file:

```env
LLM_MODEL_NAME=mistralai/Mistral-7B-v0.1
LLM_MODEL_PATH=
RAG_PERSIST_DIR=/app/rag_store
REDIS_HOST=redis
REDIS_PORT=6379
USE_WANDB=false
USE_PROMETHEUS=true
```

## API Usage

### Generate Text

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain how LoRA works for fine-tuning LLMs.",
    "max_length": 512,
    "temperature": 0.7,
    "use_cache": true
  }'
```

### RAG Query

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the best practices for risk management?",
    "k": 5,
    "use_hybrid": true
  }'
```

### Transcribe Audio

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav"
```

## Python Usage

### Using Enhanced AI Client

```python
from ai_engine.ai_client_v2 import EnhancedAIClient

# Initialize client
client = EnhancedAIClient(
    model_name="mistralai/Mistral-7B-v0.1",
    use_rag=True,
)

# Add knowledge base
client.add_knowledge_base(
    sources=["https://example.com/docs"],
    source_type="url"
)

# Request improvements (backward compatible with old ai_client)
result = client.request_improvements(context="Your code context here...")
print(result["analysis"])
```

### Using Fine-Tuned LLM Directly

```python
from ai_engine.llm.fine_tuned_model import FineTunedLLM

llm = FineTunedLLM(
    model_name="mistralai/Mistral-7B-v0.1",
    model_path="./models/fine-tuned",  # Optional
    use_quantization=True,
)

response = llm.generate(
    prompt="Your prompt here",
    max_length=512,
    temperature=0.7,
)
print(response)
```

### Using RAG System

```python
from ai_engine.rag.retrieval_system import RAGSystem
from ai_engine.llm.fine_tuned_model import FineTunedLLM

# Initialize RAG
rag = RAGSystem(
    persist_directory="./rag_store",
    vectorstore_type="faiss",
)

# Load documents
rag.load_from_urls(["https://example.com/docs"])

# Query
llm = FineTunedLLM()
# Wrap LLM for LangChain (see inference_server.py for example)
result = rag.query("Your question", llm=llm_wrapper, k=5)
print(result["result"])
```

## Fine-Tuning Guide

### Prepare Training Data

Your training data should be in JSONL format:

```jsonl
{"text": "Instruction: What is LoRA?\nResponse: LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method..."}
{"text": "Instruction: How does quantization work?\nResponse: Quantization reduces model precision from 32-bit to 4-bit..."}
```

### Fine-Tune

```bash
python scripts/train_llm.py \
    --model-name mistralai/Mistral-7B-v0.1 \
    --data-path ./data/training.jsonl \
    --output-dir ./models/fine-tuned \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4
```

### Monitor Training

If W&B is enabled, training metrics will be logged automatically. Access at https://wandb.ai

## Security Features

### Input Validation

All inputs are automatically validated:

- Length limits
- Blocked patterns (XSS, script injection)
- Profanity filtering
- Toxic language detection
- PII detection
- Prompt injection detection

### Rate Limiting

Default rate limit: 60 requests per minute per client.

Configure in `inference_server.py`:

```python
_rate_limit_max_requests = 60
_rate_limit_window = timedelta(minutes=1)
```

## Monitoring

### Prometheus Metrics

Metrics available at `http://localhost:8001/metrics`:

- `llm_inference_total`: Total inference requests
- `llm_inference_latency_seconds`: Inference latency
- `llm_inference_tokens`: Tokens generated
- `llm_model_loaded`: Model loading status

### W&B Tracking

Enable by setting `USE_WANDB=true` and configuring W&B:

```bash
wandb login
export WANDB_PROJECT="llm-inference"
```

### Grafana Dashboard

Access Grafana at `http://localhost:3000` (default credentials: admin/admin)

## Performance Optimization

### Quantization

4-bit quantization is enabled by default for faster inference and lower memory usage.

### Caching

Redis caching is enabled for frequent queries. Responses are cached for 1 hour by default.

### GPU Acceleration

Ensure CUDA is available:

```bash
nvidia-smi  # Check GPU availability
export CUDA_VISIBLE_DEVICES=0
```

## Migration from Old AI Client

The new system is backward compatible. Replace:

```python
from ai_engine.ai_client import request_improvements
```

With:

```python
from ai_engine.ai_client_v2 import request_improvements  # Same interface
```

Or use the enhanced client:

```python
from ai_engine.ai_client_v2 import EnhancedAIClient

client = EnhancedAIClient(use_rag=True)
result = client.request_improvements(context)
```

## Troubleshooting

### Model Loading Issues

- Check available GPU memory: `nvidia-smi`
- Try CPU mode by setting `CUDA_VISIBLE_DEVICES=""`
- Reduce model size (use Phi-3-mini instead of Mistral-7B)

### RAG Issues

- Ensure FAISS/ChromaDB is installed: `pip install faiss-cpu chromadb`
- Check persist directory permissions
- Verify documents are loaded: `python scripts/setup_rag.py --files test.txt`

### Memory Issues

- Enable quantization: `use_quantization=True`
- Use smaller batch sizes
- Reduce max_length in generation

## Next Steps

1. **Fine-tune on domain-specific data**: Prepare trading bot documentation and code examples
2. **Set up RAG knowledge base**: Add trading strategies, risk management docs, etc.
3. **Monitor performance**: Set up Grafana dashboards for production
4. **Scale deployment**: Use Kubernetes for auto-scaling
5. **A/B testing**: Compare fine-tuned models with base models

## Additional Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [LangChain Documentation](https://python.langchain.com/)
- [Whisper Documentation](https://github.com/openai/whisper)
- [Gradio Documentation](https://gradio.app/docs/)

