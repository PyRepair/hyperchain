from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

@dataclass
class ChainResult:
    output_dict: dict = field(default_factory=lambda: {})
    previous_result: Optional[ChainResult] = None
    next_result: Optional[ChainResult] = None

    def __post_init__(self):
        self.output_dict = deepcopy(self.output_dict)

    def __getattr__(self, name):
        if name == "output_dict":
            return self.output_dict
        if name == "previous_result":
            return self.previous_result
        if name == "next_result":
            return self.next_result
        
        if name in self.output_dict:
            return str(self.output_dict[name])
        
        return None
