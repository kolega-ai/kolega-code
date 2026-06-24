from typing import Any, Dict, Tuple

from .anthropic import ANTHROPIC_SPECS
from .moonshot import MOONSHOT_SPECS
from .kimi_coding import KIMI_CODING_SPECS
from .deepseek import DEEPSEEK_SPECS
from .openai import OPENAI_SPECS
from .openai_chatgpt import OPENAI_CHATGPT_SPECS
from .together import TOGETHER_SPECS
from .google import GOOGLE_SPECS
from .xai import XAI_SPECS
from .ollama_cloud import OLLAMA_CLOUD_SPECS
from .fireworks import FIREWORKS_SPECS
from .dashscope import DASHSCOPE_SPECS
from .zai import ZAI_SPECS

# Dictionary mapping (provider, model_name) to model specifications.
# Each entry contains context_length (maximum input tokens), max_completion_tokens,
# default_temperature, and optional model capability flags.
#
# ``supports_vision`` indicates whether a model accepts image input. When
# uncertain, default to False (the safe failure mode is a clear "model doesn't
# support images" message rather than a mid-conversation API error). The flag
# is the single tunable knob and is consumed by ``supports_vision()`` below,
# ``BaseAgent._unsupported_attachment_message`` (replacing the old hardcoded
# DeepSeek guard), and the ``read_image`` tool gate.
MODEL_SPECS: Dict[Tuple[str, str], Dict[str, Any]] = {
    **ANTHROPIC_SPECS,
    **MOONSHOT_SPECS,
    **KIMI_CODING_SPECS,
    **DEEPSEEK_SPECS,
    **OPENAI_SPECS,
    **OPENAI_CHATGPT_SPECS,
    **TOGETHER_SPECS,
    **GOOGLE_SPECS,
    **XAI_SPECS,
    **OLLAMA_CLOUD_SPECS,
    **FIREWORKS_SPECS,
    **DASHSCOPE_SPECS,
    **ZAI_SPECS,
}
