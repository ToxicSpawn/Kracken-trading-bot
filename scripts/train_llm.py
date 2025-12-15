#!/usr/bin/env python3
"""
Script to fine-tune an LLM using LoRA.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset, Dataset
from ai_engine.llm.fine_tuned_model import FineTunedLLM
from ai_engine.monitoring.tracker import MetricsTracker


def prepare_dataset(data_path: str, text_column: str = "text") -> Dataset:
    """
    Prepare dataset for training.
    Supports JSON, JSONL, CSV, or HuggingFace dataset.
    """
    data_path_obj = Path(data_path)

    if data_path_obj.suffix == ".jsonl":
        dataset = load_dataset("json", data_files=data_path, split="train")
    elif data_path_obj.suffix == ".json":
        dataset = load_dataset("json", data_files=data_path, split="train")
    elif data_path_obj.suffix == ".csv":
        dataset = load_dataset("csv", data_files=data_path, split="train")
    else:
        # Assume it's a HuggingFace dataset identifier
        dataset = load_dataset(data_path, split="train")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model name",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data (JSON, JSONL, CSV, or HF dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/fine-tuned",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per device batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name containing text in dataset",
    )

    args = parser.parse_args()

    # Initialize metrics tracker
    tracker = MetricsTracker(project_name="llm-fine-tuning", use_wandb=True)

    # Prepare dataset
    print(f"Loading dataset from: {args.data_path}")
    dataset = prepare_dataset(args.data_path, args.text_column)

    # Initialize model
    print(f"Initializing model: {args.model_name}")
    llm = FineTunedLLM(
        model_name=args.model_name,
        use_quantization=False,  # Don't quantize during training
    )

    # Fine-tune
    print("Starting fine-tuning...")
    output_dir = llm.fine_tune(
        dataset=dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print(f"Fine-tuning complete! Model saved to: {output_dir}")
    tracker.finish()


if __name__ == "__main__":
    main()

