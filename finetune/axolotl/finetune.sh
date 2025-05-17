#!/usr/bin/env bash
# Run minimal LoRA fine-tune of Qwen3-1.7B
export PYTORCH_ENABLE_MPS_FALLBACK=1
axolotl train hello_qwen3_lora.yml
