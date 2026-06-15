from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
    thinking: Optional[Any] = None
