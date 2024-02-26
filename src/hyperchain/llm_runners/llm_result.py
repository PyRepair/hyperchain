from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResult:
    output: str
    extra_llm_outputs: Optional[dict] = None
    previous_result: Optional["LLMResult"] = None
    next_result: Optional["LLMResult"] = None
