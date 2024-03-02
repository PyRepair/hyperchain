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
        formatter: Formatter = Formatter(),
        message_content_key: str = "content"
    ):
        self.input_list = input_list
        self.formatter = formatter
        self.message_content_key = message_content_key
        self.required_keys = []
        for chat_element in self.input_list:
            if self.message_content_key in chat_element:
                self.required_keys += [key for _, key, _, _ in formatter.parse(chat_element[self.message_content_key]) if key is not None]

    @classmethod
    def from_input(cls, input_list: List[dict]) -> Template[List[dict]]:
        return ChatTemplate(input_list)

    def format(self, **kwargs: Any) -> List[dict]:
        answer = []
        for chat_element in self.input_list:
            chat_element_copy = chat_element.copy()
            if self.message_content_key in chat_element_copy:
                chat_element_copy[self.message_content_key] = self.formatter.format(
                    chat_element_copy[self.message_content_key], **kwargs
                )
            answer.append(chat_element_copy)
        return answer

    def __add__(self, other: Any) -> Template[List[dict]]:
        if isinstance(other, ChatTemplate):
            return ChatTemplate(
                self.input_list + other.input_list,
                self.formatter,
            )

        if isinstance(other, list):
            return ChatTemplate(
                self.input_list + other,
                self.formatter,
            )

        raise NotImplementedError(
            "ChatTemplate only allows addition"
            "with another ChatTemplate or list of dict"
        )