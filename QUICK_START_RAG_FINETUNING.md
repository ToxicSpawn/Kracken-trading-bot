# Quick Start: RAG + Fine-Tuning (Step 1)

This guide walks through the complete RAG + Fine-Tuning workflow as described in the upgrade requirements.

## Prerequisites

Install dependencies:

```bash
pip install transformers datasets peft trl torch accelerate bitsandbytes
pip install langchain sentence-transformers faiss-cpu pypdf
```

## Step 1: Prepare Your Dataset

Create a JSONL file with prompt/response pairs:

```jsonl
{"prompt": "What is RAG?", "response": "RAG (Retrieval-Augmented Generation) combines..."}
{"prompt": "How to fine-tune an LLM?", "response": "Use LoRA (Low-Rank Adaptation)..."}
```

An example dataset is provided at `data/example_dataset.jsonl`.

## Step 2: Fine-Tune the Model

Use the complete fine-tuning script:

```bash
python scripts/finetune_with_lora.py \
    --model-name mistralai/Mistral-7B-v0.1 \
    --dataset data/example_dataset.jsonl \
    --output-dir ./fine-tuned-mistral \
    --epochs 3 \
    --batch-size 4 \
    --use-4bit
```

Options:
- `--model-name`: Base model (mistralai/Mistral-7B-v0.1, microsoft/Phi-3-mini-4k-instruct)
- `--dataset`: Path to your JSONL dataset
- `--output-dir`: Where to save the fine-tuned model
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size per device
- `--use-4bit`: Enable 4-bit quantization (saves memory)

The script will:
1. Load the base model
2. Apply LoRA configuration
3. Load and format your dataset
4. Fine-tune with 4-bit quantization
5. Save the fine-tuned model

## Step 3: Set Up RAG with Documents

Load your documents (PDFs, text files, URLs) and create a vectorstore:

```bash
python scripts/setup_rag_complete.py \
    --docs example.pdf ./docs/ README.md https://example.com/docs \
    --vectorstore-dir ./rag_vectorstore \
    --model-path ./fine-tuned-mistral \
    --query "What is RAG?"
```

The script will:
1. Load and chunk your documents
2. Create embeddings using sentence-transformers
3. Build a FAISS vectorstore
4. Create a RAG pipeline with your fine-tuned model
5. Test with an optional query

## Step 4: Query Your RAG System

Use the interactive query script:

```bash
python scripts/rag_query_interactive.py \
    --vectorstore-dir ./rag_vectorstore \
    --model-path ./fine-tuned-mistral
```

Or use it programmatically:

```python
from scripts.setup_rag_complete import create_rag_pipeline

# Create RAG pipeline
qa_chain, llm = create_rag_pipeline(
    model_path="./fine-tuned-mistral",
    vectorstore_path="./rag_vectorstore"
)

# Query
result = qa_chain({"query": "What is RAG?"})
print(result["result"])
print("\nSources:", [doc.metadata["source"] for doc in result["source_documents"]])
```

## Complete Example Workflow

```bash
# 1. Prepare dataset (or use example)
cp data/example_dataset.jsonl my_dataset.jsonl

# 2. Fine-tune model
python scripts/finetune_with_lora.py \
    --model-name mistralai/Mistral-7B-v0.1 \
    --dataset my_dataset.jsonl \
    --output-dir ./fine-tuned-mistral \
    --epochs 3 \
    --use-4bit

# 3. Set up RAG
python scripts/setup_rag_complete.py \
    --docs ./docs/ README.md \
    --vectorstore-dir ./rag_vectorstore \
    --model-path ./fine-tuned-mistral

# 4. Query interactively
python scripts/rag_query_interactive.py \
    --vectorstore-dir ./rag_vectorstore \
    --model-path ./fine-tuned-mistral
```

## Integration with Existing Code

The fine-tuned model can be used with the existing `ai_engine` infrastructure:

```python
from ai_engine.llm.fine_tuned_model import FineTunedLLM

# Load fine-tuned model
llm = FineTunedLLM(
    model_name="mistralai/Mistral-7B-v0.1",
    model_path="./fine-tuned-mistral",
    use_quantization=True,
)

# Generate text
response = llm.generate(
    prompt="What is RAG?",
    max_length=512,
    temperature=0.7,
)
```

Or use with the enhanced AI client:

```python
from ai_engine.ai_client_v2 import EnhancedAIClient

client = EnhancedAIClient(
    model_name="mistralai/Mistral-7B-v0.1",
    model_path="./fine-tuned-mistral",
    use_rag=True,
)

result = client.request_improvements(context="Your code context...")
```

## Tips

1. **Dataset Format**: Your JSONL should have either `prompt`/`response` or `instruction`/`output` fields, or a single `text` field.

2. **Memory Management**: Use `--use-4bit` to reduce memory usage. For very large models, consider using a smaller base model like Phi-3-mini.

3. **Chunk Size**: Adjust `--chunk-size` and `--chunk-overlap` based on your documents. Smaller chunks = more precise retrieval, larger chunks = more context.

4. **Fine-Tuning Tips**:
   - Start with 3 epochs
   - Use a small learning rate (2e-4)
   - Monitor training loss
   - Save checkpoints regularly

5. **RAG Tips**:
   - Include diverse documents in your knowledge base
   - Use descriptive document names for better source tracking
   - Test retrieval quality before fine-tuning
   - Adjust `k` (number of retrieved documents) based on your needs

## Troubleshooting

**Out of Memory**:
- Use `--use-4bit` flag
- Reduce batch size
- Use a smaller base model (Phi-3-mini)

**Poor Fine-Tuning Results**:
- Check dataset quality and format
- Try more epochs
- Adjust learning rate
- Ensure sufficient training examples (100+ recommended)

**RAG Not Finding Relevant Documents**:
- Check chunk size (try smaller chunks)
- Verify embeddings are loading correctly
- Test retrieval separately before using with LLM
- Consider adding more diverse documents

## Next Steps

After completing Step 1:
- **Step 2**: Containerize with Docker (see `Dockerfile` and `docker-compose.yml`)
- **Step 3**: Add multi-modal or agentic AI
- **Step 4**: Optimize costs with quantization and serverless

See `LLM_UPGRADE_GUIDE.md` for complete documentation.

