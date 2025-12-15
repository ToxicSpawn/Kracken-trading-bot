#!/bin/bash
# Startup script for the LLM inference server

set -e

# Default values
PORT=${INFERENCE_SERVER_PORT:-8000}
MODEL_NAME=${LLM_MODEL_NAME:-"mistralai/Mistral-7B-v0.1"}
MODEL_PATH=${LLM_MODEL_PATH:-""}
RAG_DIR=${RAG_PERSIST_DIR:-"./rag_store"}
REDIS_HOST=${REDIS_HOST:-"localhost"}
REDIS_PORT=${REDIS_PORT:-6379}

echo "Starting LLM Inference Server"
echo "=============================="
echo "Port: $PORT"
echo "Model: $MODEL_NAME"
if [ -n "$MODEL_PATH" ]; then
    echo "Fine-tuned model path: $MODEL_PATH"
fi
echo "RAG directory: $RAG_DIR"
echo "Redis: $REDIS_HOST:$REDIS_PORT"
echo ""

# Export environment variables
export LLM_MODEL_NAME="$MODEL_NAME"
export LLM_MODEL_PATH="$MODEL_PATH"
export RAG_PERSIST_DIR="$RAG_DIR"
export REDIS_HOST="$REDIS_HOST"
export REDIS_PORT="$REDIS_PORT"

# Create directories if they don't exist
mkdir -p "$RAG_DIR"
mkdir -p logs

# Start the server
python -m ai_engine.inference_server

