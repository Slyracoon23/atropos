import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from atropos_api.schemas import ChatMessage

logger = logging.getLogger(__name__)

model = None
tokenizer = None


def load_model(model_name: str, device: str = "mps"):
    """Load model and tokenizer. Uses float16 for MPS, bfloat16 for CUDA, float32 for CPU.
    See Qwen3 best practices.
    """
    global model, tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "mps":
        dtype = torch.float16
    elif device == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    logger.info(f"Model loaded successfully on {device}")


def chatml_messages_to_prompt(
    messages: List[ChatMessage], tokenizer, enable_thinking: bool = True
) -> str:
    """Convert a list of ChatML messages to a single prompt string, using the tokenizer's chat template if available. Pass enable_thinking as recommended in Qwen3 best practices."""
    chat_list = [{"role": m.role, "content": m.content} for m in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            chat_list,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    else:
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt
