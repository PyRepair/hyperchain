from __future__ import annotations
from typing import List, Any, Optional
from string import Formatter

from .string_template import StringTemplate

class MaskToSentinelTemplate(StringTemplate):
    def __init__(
        self,
        input_string: str,
        formatter: Formatter = Formatter(),
        mask_token: str = "<mask>",
        sentinel_token_template: str = "<extra_id_{}>",
        sentinel_start_index = 0,
        sentinel_end_index = 99,
    ):
        super().__init__(input_string, formatter)
        self.mask_token = mask_token
        self.sentinel_token_template = sentinel_token_template
        self.sentinel_start_index = sentinel_start_index
        self.sentinel_end_index = sentinel_end_index

    @classmethod
    def from_input(cls, input_string: str) -> Template[str]:
        return MaskToSentinelTemplate(input_string)

    def _apply_sentinel_tokens(self, prompt: str) -> str:
        split_prompt = prompt.split(self.mask_token)

        response = ""
        for index, prompt_part in enumerate(split_prompt):
            if index > 0:
                response += self.sentinel_token_template.format((index-1)%(self.sentinel_end_index-self.sentinel_start_index+1)+self.sentinel_start_index)
            response += prompt_part
        
        return response

    def format(self, **kwargs: Any) -> str:
        if len(kwargs) == 0:
            return self._apply_sentinel_tokens(self.input_string)
        return self._apply_sentinel_tokens(self.formatter.format(self.input_string, **kwargs))

    def __add__(self, other: Any) -> Template[str]:
        if isinstance(other, MaskToSentinelTemplate) or isinstance(StringTemplate):
            return MaskToSentinelTemplate(
                self.input_string + other.input_string,
                self.formatter,
            )
        
        if isinstance(other, str):
            return MaskToSentinelTemplate(
                self.input_string + other,
                self.formatter,
            )
        
        raise NotImplementedError(
            "MaskToSentinelTemplate only allows addition"
            "with another MaskToSentinelTemplate, StringTemplate or str"
        )