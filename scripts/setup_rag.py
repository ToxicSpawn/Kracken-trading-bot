#!/usr/bin/env python3
"""
Script to set up RAG knowledge base with documents.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.rag.retrieval_system import RAGSystem


def main():
    parser = argparse.ArgumentParser(description="Set up RAG knowledge base")
    parser.add_argument(
        "--urls",
        type=str,
        nargs="+",
        help="URLs to load documents from",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        help="File paths to load",
    )
    parser.add_argument(
        "--directories",
        type=str,
        nargs="+",
        help="Directories to load documents from",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./rag_store",
        help="Directory to persist vectorstore",
    )
    parser.add_argument(
        "--vectorstore-type",
        type=str,
        default="faiss",
        choices=["faiss", "chroma"],
        help="Vectorstore type",
    )

    args = parser.parse_args()

    # Initialize RAG system
    print(f"Initializing RAG system (type: {args.vectorstore_type})...")
    rag = RAGSystem(
        persist_directory=args.persist_dir,
        vectorstore_type=args.vectorstore_type,
    )

    # Load documents
    if args.urls:
        print(f"Loading {len(args.urls)} URLs...")
        rag.load_from_urls(args.urls)

    if args.files:
        print(f"Loading {len(args.files)} files...")
        for file_path in args.files:
            rag.load_from_file(file_path)

    if args.directories:
        print(f"Loading {len(args.directories)} directories...")
        for directory in args.directories:
            rag.load_from_directory(directory)

    print(f"RAG knowledge base setup complete! Vectorstore saved to: {args.persist_dir}")


if __name__ == "__main__":
    main()

