from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThinkingEffortSpec:
    """Model-specific thinking/effort controls and provider serialization mode."""

    options: tuple[str, ...]
    default: str
    mode: str
    budgets: dict[str, int] = field(default_factory=dict)
