from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ReasoningEffort(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ThinkingConfig:
    """Configuration for model's thinking depth"""

    budget_tokens: int = 4096


@dataclass
class GeminiThinkingConfig:
    include_thoughts: bool = True


@dataclass
class TokenCount:
    input_tokens: int
    output_tokens: Optional[int] = None


@dataclass
class GenerationParams:
    """Common parameters for text generation across providers"""

    temperature: float = 1.0
    max_completion_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[Union[ThinkingConfig, ReasoningEffort, GeminiThinkingConfig]] = None
