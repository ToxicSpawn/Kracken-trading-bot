#!/usr/bin/env python3
"""
Complete RAG setup script that combines fine-tuned models with document retrieval.
Matches the example provided in the upgrade guide.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from transformers import pipeline
from typing import Optional, List
from pathlib import Path

from ai_engine.llm.fine_tuned_model import FineTunedLLM


class FineTunedLLMWrapper(LLM):
    """Wrapper to make FineTunedLLM compatible with LangChain."""
    
    def __init__(self, llm_model: FineTunedLLM):
        super().__init__()
        self.llm_model = llm_model
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        # Remove stop tokens if provided (basic implementation)
        response = self.llm_model.generate(
            prompt=prompt,
            max_length=kwargs.get("max_length", 512),
            temperature=kwargs.get("temperature", 0.7),
            **{k: v for k, v in kwargs.items() if k not in ["max_length", "temperature"]}
        )
        return response
    
    @property
    def _llm_type(self) -> str:
        return "fine_tuned_llm"


def load_and_chunk_documents(doc_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load and chunk documents from various sources.
    
    Args:
        doc_paths: List of file paths or URLs
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of document chunks
    """
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    for doc_path in doc_paths:
        print(f"Loading: {doc_path}")
        path = Path(doc_path)
        
        if path.is_file():
            if path.suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = TextLoader(str(path))
            docs = loader.load()
        elif path.is_dir():
            loader = DirectoryLoader(str(path), glob="**/*.txt", loader_cls=TextLoader)
            docs = loader.load()
        elif doc_path.startswith("http://") or doc_path.startswith("https://"):
            loader = WebBaseLoader([doc_path])
            docs = loader.load()
        else:
            print(f"Warning: Skipping unknown path: {doc_path}")
            continue
        
        # Chunk documents
        splits = text_splitter.split_documents(docs)
        all_docs.extend(splits)
        print(f"  Loaded {len(splits)} chunks")
    
    return all_docs


def create_rag_pipeline(model_path: Optional[str], vectorstore_path: str):
    """
    Create a complete RAG pipeline with fine-tuned model.
    
    Args:
        model_path: Path to fine-tuned model (or None for base model)
        vectorstore_path: Path to FAISS vectorstore
    
    Returns:
        RAG QA chain
    """
    # Load fine-tuned model
    model_name = os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-v0.1")
    
    if model_path and Path(model_path).exists():
        print(f"Loading fine-tuned model from: {model_path}")
        llm_model = FineTunedLLM(
            model_name=model_name,
            model_path=model_path,
            use_quantization=True,
        )
    else:
        print(f"Loading base model: {model_name}")
        llm_model = FineTunedLLM(
            model_name=model_name,
            use_quantization=True,
        )
    
    # Wrap for LangChain
    llm = FineTunedLLMWrapper(llm_model)
    
    # Load embeddings
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
    )
    
    # Load vectorstore
    print(f"Loading vectorstore from: {vectorstore_path}")
    vectorstore = FAISS.load_local(vectorstore_path, embeddings)
    
    # Create RAG chain
    print("Creating RAG pipeline...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )
    
    return qa_chain, llm_model


def main():
    parser = argparse.ArgumentParser(description="Set up RAG with fine-tuned model")
    parser.add_argument(
        "--docs",
        type=str,
        nargs="+",
        required=True,
        help="Document paths (files, directories, or URLs)",
    )
    parser.add_argument(
        "--vectorstore-dir",
        type=str,
        default="./rag_vectorstore",
        help="Directory to save FAISS vectorstore",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for text splitting",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to fine-tuned model (optional)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Test query (optional)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RAG Setup with Fine-Tuned Model")
    print("=" * 60)
    print(f"Documents: {len(args.docs)} sources")
    print(f"Vectorstore: {args.vectorstore_dir}")
    print()

    # Load and chunk documents
    print("Step 1: Loading and chunking documents...")
    docs = load_and_chunk_documents(
        args.docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Total chunks: {len(docs)}")
    print()

    # Create embeddings and vectorstore
    print("Step 2: Creating embeddings and vectorstore...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
    )
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save vectorstore
    Path(args.vectorstore_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(args.vectorstore_dir)
    print(f"Vectorstore saved to: {args.vectorstore_dir}")
    print()

    # Create RAG pipeline (if model path provided or test query)
    if args.model_path or args.query:
        print("Step 3: Creating RAG pipeline...")
        qa_chain, llm = create_rag_pipeline(args.model_path, args.vectorstore_dir)
        
        # Test query
        if args.query:
            print(f"\nTesting with query: {args.query}")
            print("-" * 60)
            result = qa_chain({"query": args.query})
            print("Answer:")
            print(result["result"])
            print("\nSources:")
            for i, doc in enumerate(result.get("source_documents", [])[:3], 1):
                print(f"{i}. {doc.metadata.get('source', 'unknown')}")
                print(f"   {doc.page_content[:200]}...")
        else:
            print("\nRAG pipeline ready! Use --query to test.")
    
    print("\n" + "=" * 60)
    print("RAG setup complete!")
    print(f"Vectorstore: {args.vectorstore_dir}")
    print("\nTo use in code:")
    print("  from scripts.setup_rag_complete import create_rag_pipeline")
    print(f"  qa_chain, _ = create_rag_pipeline('{args.model_path}', '{args.vectorstore_dir}')")
    print("  result = qa_chain({'query': 'Your question here'})")
    print("=" * 60)


if __name__ == "__main__":
    main()

