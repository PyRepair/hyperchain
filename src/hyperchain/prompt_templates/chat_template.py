from __future__ import annotations
from typing import List, Any, Optional
from string import Formatter

from .base_template import Template

class ChatTemplate(Template[List[dict]]):
    input_list: List[dict]
    formatter: Formatter

    def __init__(
        self,
        input_list: List[dict],
        input_variables: Optional[List[str]] = None,
        formatter: Formatter = Formatter(),
    ):
        super().__init__(input_variables)
        self.input_list = input_list
        self.formatter = formatter

    @classmethod
    def from_input(cls, input_list: List[dict]) -> Template[List[dict]]:
        return ChatTemplate(input_list)

    def _format(self, **kwargs: Any) -> List[dict]:
        answer = []
        for chat_element in self.input_list:
            chat_element_copy = chat_element.copy()
            if "content" in chat_element_copy:
                chat_element_copy["content"] = self.formatter.format(
                    chat_element_copy["content"], **kwargs
                )
            answer.append(chat_element_copy)
        return answer

    def __add__(self, other: Any) -> Template[List[dict]]:
        if isinstance(other, ChatTemplate):
            return ChatTemplate(
                self.input_list + other.input_list,
                self.input_variables + other.input_variables,
                self.formatter,
            )

        if isinstance(other, list):
            return ChatTemplate(
                self.input_list + other,
                self.input_variables,
                self.formatter,
            )

        raise NotImplementedError(
            "ChatTemplate only allows addition"
            "with another ChatTemplate or list of dict"
        )