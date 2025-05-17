"""
fine_tune_lora_mps.py
=====================
A minimal example of fine‑tuning **Qwen3‑1.7B** with **LoRA** on Apple Silicon (M‑series) using PyTorch **MPS** backend and Hugging Face libraries (`transformers`, `datasets`, `peft`).

▸ Tested on macOS 14 / Python 3.11 / PyTorch 2.2 nightly.
▸ Uses the 1.7‑billion‑parameter *Qwen/Qwen3‑1.7B* dense model, which fits into ≈6 GB VRAM fp16.

Install requirements:

```bash
pip install --upgrade "torch>=2.2" "transformers>=4.52" datasets peft accelerate
# Apple Silicon users may prefer nightly wheels:
# pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

Run fine‑tuning (single GPU):

```bash
python fine_tune_lora_mps.py
```
"""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# ---------------------------------------------------------------------------
# Configuration – edit these values to taste --------------------------------
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-1.7B"  # updated to match YAML
DATASET_PATH = "mlabonne/FineTome-100k"  # unchanged
DATASET_SPLIT = "train[:5%]"  # updated to match YAML
OUTPUT_DIR = Path("./outputs/out")  # updated to match YAML
MAX_SEQ_LEN = 2048  # updated to match YAML
BATCH_SIZE = 1  # micro_batch_size in YAML
GR_ACCUM = 2  # gradient_accumulation_steps in YAML
EPOCHS = 1  # unchanged
LEARNING_RATE = 2e-4  # 0.0002 in YAML
LORA_R = 16  # updated to match YAML
LORA_ALPHA = 32  # updated to match YAML
LORA_DROPOUT = 0.05  # unchanged (not specified in YAML)
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "down_proj",
    "up_proj",
]  # unchanged
WEIGHT_DECAY = 0.0  # added from YAML
WARMUP_STEPS = 10  # added from YAML
LOGGING_STEPS = 1  # updated to match YAML
SAVE_STEPS = None  # will set to save only at end of training
BF16 = True  # bf16: auto in YAML (set True if available)
# FLASH_ATTENTION, lora_mlp_kernel, lora_qkv_kernel, lora_o_kernel, adapter: not directly supported in HF Trainer+PEFT
# Gradient checkpointing: enable if possible
# ---------------------------------------------------------------------------


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load tokenizer & base model ---------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # Trainer expects a pad token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=(
            torch.bfloat16
            if BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        ),
        device_map={"": device},
    )

    # 2. Prepare LoRA PEFT wrapper -----------------------------------------
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # If you later want 4‑bit quantisation, uncomment next line (CPU only)
    # model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 3. Load & tokenise dataset -------------------------------------------
    dataset = load_dataset(DATASET_PATH, split=DATASET_SPLIT)

    def tokenize(example):
        # FineTome provides the conversation template in "text" column.
        text = example.get("text") or ""
        # Use Qwen3 chat template for alignment
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = tokenizer(
            formatted,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenised = dataset.map(tokenize, remove_columns=dataset.column_names)

    # 4. TrainingArguments & Trainer ---------------------------------------
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GR_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        optim=(
            "adamw_torch" if hasattr(torch.optim, "AdamW") else "adamw_hf"
        ),  # adamw_torch_4bit not available, fallback
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        save_steps=100,  # effectively disables intermediate saves
        bf16=BF16,
        fp16=not BF16,  # mutually exclusive
        report_to="none",
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        gradient_checkpointing=True,  # enable if possible
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenised,
    )

    # 5. Train --------------------------------------------------------------
    trainer.train()

    # 6. Save adapter & tokenizer ------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
