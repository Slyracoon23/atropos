#!/usr/bin/env python
# Transformers model server with FastAPI that mimics vLLM's interface

import argparse
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import uvicorn
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize global variables for model and tokenizer
model = None
tokenizer = None

# Initialize the app
app = FastAPI(title="Transformers Model Server")
router = APIRouter(prefix="/v1")


class GenerateRequest(BaseModel):
    prompts: List[str]
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 16
    guided_decoding_regex: Optional[str] = None


class GenerateResponse(BaseModel):
    completions: List[str]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 16
    guided_decoding_regex: Optional[str] = None


def load_model(model_name: str, device: str = "mps"):
    """Load model and tokenizer."""
    global model, tokenizer

    logger.info(f"Loading model: {model_name}")

    # Load tokenizer with padding token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate configuration and dtype based on device
    if device == "mps":
        # MPS typically works better with float16 than bfloat16
        dtype = torch.float16
    elif device == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )

    logger.info(f"Model loaded successfully on {device}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context for FastAPI app to load model on startup."""
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-1.7B")
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    load_model(model_name, device)
    yield
    # No cleanup needed


app = FastAPI(title="Transformers Model Server", lifespan=lifespan)
router = APIRouter(prefix="/v1")


@router.get("/health")
async def health():
    """Health check endpoint."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


def chatml_messages_to_prompt(messages: List[ChatMessage], tokenizer) -> str:
    """Convert a list of ChatML messages to a single prompt string, using the tokenizer's chat template if available."""
    chat_list = [{"role": m.role, "content": m.content} for m in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            chat_list, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt


@router.post("/completions", response_model=GenerateResponse)
async def generate(
    request: dict,  # Accept raw dict to support both formats
    background_tasks: BackgroundTasks,
):
    """Generate completions for the provided prompts or messages."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Determine if this is a ChatML-style request or legacy prompt request
    if "messages" in request:
        # Parse as ChatML
        chat_req = ChatCompletionRequest(**request)
        prompt = chatml_messages_to_prompt(chat_req.messages, tokenizer)
        n = chat_req.n
        repetition_penalty = chat_req.repetition_penalty
        temperature = chat_req.temperature
        top_p = chat_req.top_p
        top_k = chat_req.top_k
        max_tokens = chat_req.max_tokens
    else:
        # Legacy prompt-based
        gen_req = GenerateRequest(**request)
        # For compatibility, just use the first prompt
        prompt = gen_req.prompts[0]
        n = gen_req.n
        repetition_penalty = gen_req.repetition_penalty
        temperature = gen_req.temperature
        top_p = gen_req.top_p
        top_k = gen_req.top_k
        max_tokens = gen_req.max_tokens

    # Prepare generation config
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else None,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    all_completions = []
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    for _ in range(n):
        with torch.no_grad():
            output = model.generate(
                input_ids,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=False,
            )
        completion_tokens = output.sequences[0, input_ids.shape[1] :].tolist()
        completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        all_completions.append(completion_text)

    return {"completions": all_completions}


@router.get("/get_world_size/")
async def get_world_size():
    """Get model parallelism info (simplified for transformers)."""
    return {"world_size": 1}  # No parallelism in this implementation


@router.post("/reset_prefix_cache/")
async def reset_prefix_cache():
    """Clear past key/value caches (not needed for transformers but kept for interface compatibility)."""
    return {"message": "Cache reset successful"}


app.include_router(router)


def main():
    parser = argparse.ArgumentParser(
        description="Transformers model server with FastAPI"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.7B", help="Model name or path"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=9001, help="Port to run the server on"
    )

    # Device selection with MPS support
    default_device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Device to run the model on (mps, cuda, cpu)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Log level (debug, info, warning, error, critical)",
    )

    args = parser.parse_args()

    # Set environment variable for model loading in startup event
    os.environ["MODEL_NAME"] = args.model

    # Start the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
