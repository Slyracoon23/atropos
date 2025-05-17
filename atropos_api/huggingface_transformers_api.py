#!/usr/bin/env python
# Transformers model server with FastAPI that mimics vLLM's interface

import argparse
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict

import torch
import uvicorn
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException

from atropos_api.model_utils import chatml_messages_to_prompt
from atropos_api.schemas import ChatCompletionRequest, GenerateRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global model/tokenizer
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context for FastAPI app to load model on startup."""
    global model, tokenizer
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-1.7B")
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")

    # Import these after FastAPI app is created to avoid circular imports
    from atropos_api.model_utils import load_model

    load_model(model_name, device)

    from atropos_api.model_utils import model as loaded_model
    from atropos_api.model_utils import tokenizer as loaded_tokenizer

    model = loaded_model
    tokenizer = loaded_tokenizer
    yield


# Create FastAPI app first
app = FastAPI(title="Transformers Model Server", lifespan=lifespan)

# Create router after app
router = APIRouter()


@router.get("/health")
async def health():
    """Health check endpoint."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@router.post("/completions")
async def generate(
    request: dict,
    background_tasks: BackgroundTasks,
):
    """Generate completions for the provided prompts or messages. Uses best-practice defaults for Qwen3 (see model card)."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Incoming request: {request}")

    if "messages" in request:
        chat_req = ChatCompletionRequest(**request)
        prompt = chatml_messages_to_prompt(
            chat_req.messages, tokenizer, enable_thinking=chat_req.enable_thinking
        )
        n = chat_req.n
        repetition_penalty = chat_req.repetition_penalty
        temperature = chat_req.temperature
        top_p = chat_req.top_p
        top_k = chat_req.top_k
        max_tokens = chat_req.max_tokens
        presence_penalty = chat_req.presence_penalty
    else:
        gen_req = GenerateRequest(**request)
        prompt = gen_req.prompts[0]
        n = gen_req.n
        repetition_penalty = gen_req.repetition_penalty
        temperature = gen_req.temperature
        top_p = gen_req.top_p
        top_k = gen_req.top_k
        max_tokens = gen_req.max_tokens
        presence_penalty = gen_req.presence_penalty

    # Input validation
    if n < 1:
        raise HTTPException(status_code=400, detail="n must be >= 1")
    if not (1 <= max_tokens <= 32768):
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be between 1 and 32768 (Qwen3 context limit)",
        )

    from transformers import GenerationConfig

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
    if presence_penalty is not None:
        gen_config.presence_penalty = presence_penalty

    all_completions = []
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    # Batch generation if n > 1 and prompt is the same
    if n > 1:
        input_ids_batch = input_ids.expand(n, -1)
        with torch.no_grad():
            output = model.generate(
                input_ids_batch,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=False,
            )
        for i in range(n):
            completion_tokens = output.sequences[i, input_ids.shape[1] :].tolist()
            completion_text = tokenizer.decode(
                completion_tokens, skip_special_tokens=True
            )
            all_completions.append(completion_text)
    else:
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

    logger.info(f"Generated completions: {all_completions}")
    from atropos_api.schemas import GenerateResponse

    return GenerateResponse(completions=all_completions)


@router.get("/get_world_size")
async def get_world_size() -> Dict[str, int]:
    """Get model parallelism info (simplified for transformers)."""
    return {"world_size": 1}


@router.post("/reset_prefix_cache")
async def reset_prefix_cache() -> Dict[str, str]:
    """Clear past key/value caches (not needed for transformers but kept for interface compatibility)."""
    return {"message": "Cache reset successful"}


# Mount the router with prefix AFTER all endpoints are defined
app.include_router(router, prefix="/v1")

# Debug route registration
print("All registered routes:")
for route in app.routes:
    print(f"Route: {route.path}, Methods: {route.methods}")


def main():
    """Entrypoint for the model server. Loads model and starts FastAPI app. See Qwen3 best practices for recommended parameters."""
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
    os.environ["MODEL_NAME"] = args.model
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
