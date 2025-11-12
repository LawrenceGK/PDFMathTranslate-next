"""Configuration for supported translation models."""

from typing import TypedDict


class ModelInfo(TypedDict):
    """Information about a translation model."""
    id: str
    name: str
    provider: str
    description: str
    context_length: int


# Supported models via OpenAI Compatible API
SUPPORTED_MODELS: list[ModelInfo] = [
    {
        "id": "DeepSeek-V3.2-Exp",
        "name": "DeepSeek V3.2",
        "provider": "DeepSeek",
        "description": "Advanced reasoning model from DeepSeek",
        "context_length": 64000,
    },
    {
        "id": "gpt-5-mini",
        "name": "GPT 5 Mini",
        "provider": "OpenAI",
        "description": "Latest GPT-5 mini model",
        "context_length": 128000,
    },
    {
        "id": "qwen3-vl-235b-a22b-instruct",
        "name": "Qwen3 235B",
        "provider": "Alibaba",
        "description": "Large Qwen3 model with vision capabilities",
        "context_length": 32768,
    },
    {
        "id": "Kimi-K2-0905",
        "name": "Kimi K2",
        "provider": "Moonshot",
        "description": "Kimi K2 intelligent model",
        "context_length": 128000,
    },
    {
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5",
        "provider": "Anthropic",
        "description": "Latest Claude Haiku model",
        "context_length": 200000,
    },
]

# Default model if none specified
DEFAULT_MODEL = "DeepSeek-V3.2-Exp"


def get_model_by_id(model_id: str) -> ModelInfo | None:
    """Get model information by ID."""
    for model in SUPPORTED_MODELS:
        if model["id"] == model_id:
            return model
    return None


def is_model_supported(model_id: str) -> bool:
    """Check if a model is supported."""
    return get_model_by_id(model_id) is not None
