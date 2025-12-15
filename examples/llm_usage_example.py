"""
Example usage of the enhanced LLM system.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.ai_client_v2 import EnhancedAIClient
from ai_engine.llm.fine_tuned_model import FineTunedLLM
from ai_engine.rag.retrieval_system import RAGSystem


def example_enhanced_client():
    """Example using the enhanced AI client."""
    print("=" * 60)
    print("Example: Using Enhanced AI Client")
    print("=" * 60)

    # Initialize client
    client = EnhancedAIClient(
        model_name="mistralai/Mistral-7B-v0.1",  # or use env var LLM_MODEL_NAME
        use_rag=True,
    )

    # Add knowledge base (optional)
    # client.add_knowledge_base(
    #     sources=["https://example.com/docs"],
    #     source_type="url"
    # )

    # Use backward-compatible interface
    context = """
    Code review context:
    - Trading bot with risk management
    - Current PnL tracking implementation
    - Need to improve error handling
    """
    result = client.request_improvements(context)

    print("Analysis:", result.get("analysis"))
    print("\nRisks:", result.get("risks"))
    print("\nRecommended Changes:", result.get("recommended_changes"))


def example_direct_llm():
    """Example using LLM directly."""
    print("\n" + "=" * 60)
    print("Example: Using Fine-Tuned LLM Directly")
    print("=" * 60)

    # Initialize LLM
    llm = FineTunedLLM(
        model_name="mistralai/Mistral-7B-v0.1",
        use_quantization=True,  # Use 4-bit quantization for faster inference
    )

    # Generate text
    prompt = "Explain how LoRA (Low-Rank Adaptation) works for fine-tuning large language models."
    response = llm.generate(
        prompt=prompt,
        max_length=512,
        temperature=0.7,
    )

    print(f"Prompt: {prompt}")
    print(f"\nResponse: {response}")


def example_rag_system():
    """Example using RAG system."""
    print("\n" + "=" * 60)
    print("Example: Using RAG System")
    print("=" * 60)

    # Initialize RAG system
    rag = RAGSystem(
        persist_directory="./rag_store",
        vectorstore_type="faiss",
    )

    # Load documents (example)
    # rag.load_from_urls(["https://example.com/docs"])
    # rag.load_from_file("./docs/README.md")
    # rag.load_from_directory("./docs")

    # Search for relevant documents
    query = "What are the best practices for risk management in trading?"
    docs = rag.similarity_search(query, k=3)

    print(f"Query: {query}")
    print(f"\nFound {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n{i}. Source: {doc.metadata.get('source', 'unknown')}")
        print(f"   Content: {doc.page_content[:200]}...")


if __name__ == "__main__":
    import os

    # Check if we're in a suitable environment
    if not os.getenv("LLM_MODEL_NAME"):
        print("Note: Set LLM_MODEL_NAME environment variable for custom models")
        print("Example: export LLM_MODEL_NAME='mistralai/Mistral-7B-v0.1'")
        print()

    try:
        # Run examples
        # Uncomment the examples you want to run:

        # example_enhanced_client()
        # example_direct_llm()
        # example_rag_system()

        print("\nUncomment the example functions in the code to run them.")
        print("Make sure you have:")
        print("1. Installed all dependencies: pip install -r requirements.txt")
        print("2. Set LLM_MODEL_NAME environment variable (optional)")
        print("3. Have sufficient GPU memory or use CPU mode")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")

