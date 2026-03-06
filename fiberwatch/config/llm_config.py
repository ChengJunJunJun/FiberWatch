"""
LLM configuration for FiberWatch intelligent analysis.

Supports Qwen (通义千问) via OpenAI-compatible API.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class LLMConfig:
    """Configuration for LLM-based analysis."""

    api_key: str = "sk-8aac6817d59b4ec18389c2433c7e4604"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-max"
    temperature: float = 0.3
    max_tokens: int = 2048

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError(
                "API key is required. Set DASHSCOPE_API_KEY environment variable "
                "or pass api_key parameter."
            )


def load_llm_config(
    config_path: Optional[Union[str, Path]] = None,
    **overrides,
) -> LLMConfig:
    """Load LLM config from JSON file with optional overrides."""
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f).get("llm", {})
        config = LLMConfig(**data)
    else:
        config = LLMConfig()

    for key, value in overrides.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    return config
