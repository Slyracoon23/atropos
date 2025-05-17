# ===========================================================================
# Fine-tuning Qwen3-1.7B with LoRA using TRL's SFTTrainer on macOS MPS
# ===========================================================================

# Imports -------------------------------------------------------------------
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Device setup --------------------------------------------------------------
mps_available = hasattr(torch, "mps") and torch.backends.mps.is_available()
device = torch.device("mps") if mps_available else torch.device("cpu")
print(f"Using device: {device}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration -------------------------------------------------------------
model_id = "Qwen/Qwen3-1.7B"

# Dataset configuration
# You can replace this with another dataset like "mlabonne/guanaco-llama2-1k"
dataset_name = "tatsu-lab/alpaca"

# Output directory
output_dir = Path("./qwen3-1.7b-finetuned")

# LoRA configuration --------------------------------------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
)

# Training configuration ----------------------------------------------------
training_args = SFTConfig(
    output_dir=str(output_dir),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=500,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=100,
    # use_mps_device=mps_available, # no longer needed
    fp16=False,
    bf16=False,
    optim="adamw_torch",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    seed=42,
    neftune_noise_alpha=5,
    max_seq_length=512,
    dataset_text_field="text",
)


# Main routine --------------------------------------------------------------
def main():
    print("Loading dataset...")
    dataset = load_dataset(dataset_name, split="train[:1000]")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    print("Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model()
    print("Training complete!")


if __name__ == "__main__":
    main()
