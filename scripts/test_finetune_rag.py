#!/usr/bin/env python3
"""
Quick test script to verify fine-tuning and RAG setup works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig
        from trl import SFTTrainer
        from datasets import load_dataset
        print("✓ transformers, peft, trl, datasets")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        return False
    
    try:
        from langchain.document_loaders import PyPDFLoader, WebBaseLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
        print("✓ langchain, sentence-transformers, faiss-cpu")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        return False
    
    try:
        from ai_engine.llm.fine_tuned_model import FineTunedLLM
        print("✓ ai_engine.llm.fine_tuned_model")
    except ImportError as e:
        print(f"✗ Missing ai_engine module: {e}")
        return False
    
    return True


def test_dataset_format():
    """Test dataset format."""
    print("\nTesting dataset format...")
    dataset_path = Path(__file__).parent.parent / "data" / "example_dataset.jsonl"
    
    if not dataset_path.exists():
        print(f"✗ Dataset not found: {dataset_path}")
        return False
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        print(f"✓ Dataset loaded: {len(dataset)} examples")
        
        # Check format
        example = dataset[0]
        print(f"✓ Example keys: {list(example.keys())}")
        return True
    except Exception as e:
        print(f"✗ Dataset error: {e}")
        return False


def main():
    print("=" * 60)
    print("Fine-Tuning + RAG Setup Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n✗ Import test failed. Install missing packages:")
        print("  pip install transformers peft trl datasets")
        print("  pip install langchain sentence-transformers faiss-cpu pypdf")
        return False
    
    # Test dataset
    if not test_dataset_format():
        print("\n✗ Dataset test failed")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("\nNext steps:")
    print("1. Fine-tune: python scripts/finetune_with_lora.py --dataset data/example_dataset.jsonl --output-dir ./fine-tuned-mistral")
    print("2. Setup RAG: python scripts/setup_rag_complete.py --docs README.md --vectorstore-dir ./rag_vectorstore")
    print("3. Query: python scripts/rag_query_interactive.py --vectorstore-dir ./rag_vectorstore")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

