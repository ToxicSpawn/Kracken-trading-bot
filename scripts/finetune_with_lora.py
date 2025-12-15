#!/usr/bin/env python3
"""
Complete fine-tuning script with LoRA for Mistral-7B or Phi-3.
Matches the example provided in the upgrade guide.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch

def prepare_dataset_from_jsonl(data_path: str):
    """Load dataset from JSONL file."""
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # Format dataset for instruction following (if needed)
    def format_prompt(example):
        # If dataset has prompt/response format, combine them
        if "prompt" in example and "response" in example:
            text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"
        elif "instruction" in example and "output" in example:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        elif "text" in example:
            text = example["text"]
        else:
            # Use first non-meta field as text
            text = str(example)
        
        return {"text": text}
    
    dataset = dataset.map(format_prompt, remove_columns=[col for col in dataset.column_names if col != "text"])
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model name (e.g., mistralai/Mistral-7B-v0.1, microsoft/Phi-3-mini-4k-instruct)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training dataset (JSONL format)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./fine-tuned-mistral",
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
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Fine-tuning LLM with LoRA")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"4-bit quantization: {args.use_4bit}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with optional 4-bit quantization
    print("Loading model...")
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto" if args.use_4bit else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(f"LoRA configured: r={args.lora_r}, alpha={args.lora_alpha}")

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = prepare_dataset_from_jsonl(args.dataset)
    print(f"Dataset loaded: {len(dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_steps=100,
        logging_steps=10,
        fp16=torch.cuda.is_available() and not args.use_4bit,
        bf16=False,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        report_to="none",  # Change to "wandb" if you want W&B logging
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        packing=False,
    )

    # Train
    print("Starting training...")
    print("-" * 60)
    trainer.train()

    # Save model
    print("-" * 60)
    print(f"Saving fine-tuned model to {args.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")

    print("\n" + "=" * 60)
    print("Next steps:")
    print(f"1. Use the model: --model-path {args.output_dir}")
    print("2. Load in your code:")
    print(f"   from ai_engine.llm.fine_tuned_model import FineTunedLLM")
    print(f"   llm = FineTunedLLM(model_path='{args.output_dir}')")
    print("=" * 60)


if __name__ == "__main__":
    main()

