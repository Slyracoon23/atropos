from typing import List, Optional

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Request for text generation. See Qwen3 best practices for recommended parameters."""

    prompts: List[str]
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 16
    guided_decoding_regex: Optional[str] = None
    presence_penalty: Optional[float] = None  # Qwen3: helps reduce repetition


class GenerateResponse(BaseModel):
    completions: List[str]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 0.6  # Best practice for thinking mode (Qwen3 model card)
    top_p: float = 0.95  # Best practice for thinking mode (Qwen3 model card)
    top_k: int = 20  # Best practice for thinking mode (Qwen3 model card)
    min_p: float = 0.0
    max_tokens: int = 16
    guided_decoding_regex: Optional[str] = None
    enable_thinking: bool = True  # Expose enable_thinking (Qwen3 model card)
    presence_penalty: Optional[float] = None  # Qwen3: helps reduce repetition
