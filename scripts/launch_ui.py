#!/usr/bin/env python3
"""
Launch Gradio UI for LLM inference.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.ui.gradio_app import launch_ui


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio UI")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (defaults to env var LLM_MODEL_NAME)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to fine-tuned model (defaults to env var LLM_MODEL_PATH)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run UI on",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG",
    )

    args = parser.parse_args()

    launch_ui(
        model_name=args.model_name,
        model_path=args.model_path,
        enable_rag=not args.no_rag,
        port=args.port,
    )


if __name__ == "__main__":
    main()

