from __future__ import annotations
from typing import List, Any, Optional
from string import Formatter

from .base_template import Template

class StringTemplate(Template[str]):
    input_string: str
    formatter: Formatter

    def __init__(
        self,
        input_string: str,
        formatter: Formatter = Formatter(),
    ):
        self.input_string = input_string
        self.formatter = formatter
        self.required_keys = [key for _, key, _, _ in formatter.parse(input_string) if key is not None]
    
    @classmethod
    def from_input(cls, input_string: str) -> StringTemplate:
        return StringTemplate(input_string)

    def to_file(self, file_name: str):
        file_to_write = open(file_name, "w")
        file_to_write.write(self.input_string)
        file_to_write.close()
    
    @classmethod
    def from_file(cls, file_name: str):
        file_to_write = open(file_name, "r")
        return cls(file_to_write.read())
        file_to_write.close()

    def format(self, **kwargs: Any) -> str:
        if len(kwargs) == 0:
            return self.input_string
        return self.formatter.format(self.input_string, **kwargs)

    def __add__(self, other: Any) -> Template[str]:
        if isinstance(other, StringTemplate):
            return StringTemplate(
                self.input_string + other.input_string,
                self.formatter,
            )
        if isinstance(other, str):
            return StringTemplate(
                self.input_string + other,
                self.formatter,
            )
        raise NotImplementedError(
            "StringTemplate only allows addition"
            "with another StringTemplate or str"
        )
