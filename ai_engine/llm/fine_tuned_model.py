"""
Fine-tuned LLM module using LoRA for efficient adaptation.
Supports Mistral-7B, Llama-3-8B, and Phi-3 models.
"""
import os
import logging
import torch
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import SFTTrainer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FineTunedLLM:
    """
    Fine-tuned LLM with LoRA support for efficient domain adaptation.
    Supports quantization for faster inference.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        model_path: Optional[str] = None,
        use_quantization: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the fine-tuned LLM.

        Args:
            model_name: Base model name (mistralai/Mistral-7B-v0.1, meta-llama/Llama-3-8B, microsoft/Phi-3-mini-4k-instruct)
            model_path: Path to fine-tuned LoRA adapter (optional)
            use_quantization: Use 4-bit quantization for faster inference
            lora_config: Custom LoRA configuration
            device: Device to run on (cuda, cpu, auto)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers, peft, and trl packages are required. "
                "Install with: pip install transformers peft trl bitsandbytes accelerate"
            )

        self.model_name = model_name
        self.model_path = Path(model_path) if model_path else None
        self.use_quantization = use_quantization and torch.cuda.is_available()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Default LoRA config
        self.default_lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        if lora_config:
            self.default_lora_config.update(lora_config)

        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.model_name} on device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup quantization config if enabled
        quantization_config = None
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization for faster inference")

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.use_quantization else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        # Load LoRA adapter if provided
        if self.model_path and self.model_path.exists():
            logger.info(f"Loading LoRA adapter from: {self.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
        elif not self.use_quantization:
            # Apply LoRA to base model if no adapter exists and not quantized
            lora_config_obj = LoraConfig(**self.default_lora_config)
            self.model = get_peft_model(self.model, lora_config_obj)
            logger.info("Applied LoRA to base model")

        if not self.use_quantization:
            self.model = self.model.to(self.device)
            self.model.eval()

        logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if not self.use_quantization:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def fine_tune(
        self,
        dataset,
        output_dir: str = "./results",
        num_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        save_steps: int = 500,
        logging_steps: int = 100,
        **kwargs
    ):
        """
        Fine-tune the model using LoRA.

        Args:
            dataset: Training dataset (HuggingFace dataset or compatible)
            output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            save_steps: Steps between checkpoints
            logging_steps: Steps between logging
            **kwargs: Additional training arguments
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Apply LoRA if not already applied
        if not isinstance(self.model, (PeftModel, get_peft_model)):
            lora_config_obj = LoraConfig(**self.default_lora_config)
            self.model = get_peft_model(self.model, lora_config_obj)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            fp16=self.device == "cuda",
            bf16=False,
            optim="paged_adamw_8bit",
            report_to="none",  # Can be changed to "wandb" or "tensorboard"
            **kwargs
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            args=training_args,
            tokenizer=self.tokenizer,
            packing=False,
        )

        # Train
        logger.info("Starting fine-tuning...")
        trainer.train()

        # Save
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Fine-tuning complete. Model saved to: {output_dir}")

        return output_dir

