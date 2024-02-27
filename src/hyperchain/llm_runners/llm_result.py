from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any

from copy import deepcopy

@dataclass
class LLMResult:
    output: str
    extra_llm_outputs: Optional[dict] = field(default_factory=lambda: {})

    def __post_init__(self):
        self.extra_llm_outputs = deepcopy(self.extra_llm_outputs)

    def replace(self, old:str, new:str):
        return LLMResult(self.output.replace(old, new), self.extra_llm_outputs)
    
    def __add__(self, other: Any):
        if isinstance(other, LLMResult):
            return LLMResult(self.output + other.output, self.extra_llm_outputs|other.extra_llm_outputs)
        
        return LLMResult(self.output + other, self.extra_llm_outputs)

    def __str__(self):
        return self.output
