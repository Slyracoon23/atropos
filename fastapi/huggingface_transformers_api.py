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
    completion_ids: List[List[int]]


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


@router.post("/completions", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate completions for the provided prompts."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Prepare generation config
    gen_config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        do_sample=True if request.temperature > 0 else False,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k if request.top_k > 0 else None,
        repetition_penalty=request.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    all_completion_ids = []

    # Process each prompt
    for prompt in request.prompts:
        prompt_completion_ids = []

        # Tokenize the prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        # Generate completions
        for _ in range(request.n):
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    generation_config=gen_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                )

            # Extract only the new tokens (remove input)
            completion_tokens = output.sequences[0, input_ids.shape[1] :].tolist()
            prompt_completion_ids.append(completion_tokens)

        all_completion_ids.extend(prompt_completion_ids)

    return {"completion_ids": all_completion_ids}


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
