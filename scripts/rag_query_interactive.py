#!/usr/bin/env python3
"""
Interactive RAG query script using fine-tuned model.
Allows you to query your RAG system interactively.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from ai_engine.llm.fine_tuned_model import FineTunedLLM

# Define wrapper here to avoid circular imports
class FineTunedLLMWrapper(LLM):
    """Wrapper to make FineTunedLLM compatible with LangChain."""
    
    def __init__(self, llm_model: FineTunedLLM):
        super().__init__()
        self.llm_model = llm_model
    
    def _call(self, prompt: str, stop=None, **kwargs):
        response = self.llm_model.generate(
            prompt=prompt,
            max_length=kwargs.get("max_length", 512),
            temperature=kwargs.get("temperature", 0.7),
        )
        return response
    
    @property
    def _llm_type(self) -> str:
        return "fine_tuned_llm"


def main():
    parser = argparse.ArgumentParser(description="Interactive RAG query")
    parser.add_argument(
        "--vectorstore-dir",
        type=str,
        default="./rag_vectorstore",
        required=True,
        help="Path to FAISS vectorstore",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to fine-tuned model (optional)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Base model name (defaults to env var)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Interactive RAG Query")
    print("=" * 60)
    print(f"Vectorstore: {args.vectorstore_dir}")
    if args.model_path:
        print(f"Fine-tuned model: {args.model_path}")
    print()

    # Load model
    print("Loading model...")
    model_name = args.model_name or os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-v0.1")
    
    if args.model_path and Path(args.model_path).exists():
        llm_model = FineTunedLLM(
            model_name=model_name,
            model_path=args.model_path,
            use_quantization=True,
        )
    else:
        llm_model = FineTunedLLM(
            model_name=model_name,
            use_quantization=True,
        )
    
    llm = FineTunedLLMWrapper(llm_model)
    print("Model loaded!")

    # Load vectorstore
    print("Loading vectorstore...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.load_local(args.vectorstore_dir, embeddings)
    print("Vectorstore loaded!")

    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )

    print("\n" + "=" * 60)
    print("Ready! Enter your questions (or 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break
            
            if not query:
                continue
            
            print("\nSearching and generating answer...")
            result = qa_chain({"query": query})
            
            print("\n" + "-" * 60)
            print("Answer:")
            print(result["result"])
            
            if result.get("source_documents"):
                print("\nSources:")
                for i, doc in enumerate(result["source_documents"][:3], 1):
                    source = doc.metadata.get("source", "unknown")
                    content = doc.page_content[:150].replace("\n", " ")
                    print(f"{i}. {source}")
                    print(f"   {content}...")
            
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

