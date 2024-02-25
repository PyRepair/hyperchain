from __future__ import annotations
from abc import ABC, abstractmethod
from string import Formatter
from typing import List, Generic, TypeVar, Any, Optional
import pickle

T = TypeVar("T")


class Template(Generic[T], ABC):
    input_variables: Optional[List[str]]

    def __init__(self, input_variables=None):
        if input_variables is not None:
            invalid_input_variables = [
                ivar for ivar in input_variables if not ivar.isidentifier()
            ]
            if len(invalid_input_variables) > 0:
                raise ValueError(
                    "Invalid input variable names provided:"
                    f" {invalid_input_variables}"
                )
        self.input_variables = input_variables

    @classmethod
    @abstractmethod
    def from_input(cls, input: T) -> Template[T]:
        """
        Load template directly from specified input type
        """

    @classmethod
    @abstractmethod
    def from_file(cls, file_to_read: file) -> Template[T]:
        """
        Load template from file
        """

    def to_file(self, file_name: str):
        """
        Save template to file
        """

    @abstractmethod
    def _format(self, **kwargs: Any) -> str:
        """
        Format string to be passed to the model
        """

    def format(self, **kwargs: Any) -> str:
        if self.input_variables is None:
            return self._format(**kwargs)

        missing_input_variables = [
            arg for arg in self.input_variables if not arg in kwargs.keys()
        ]
        if len(missing_input_variables) == 0:
            return self._format(**kwargs)

        raise ValueError(
            "Following required input variables weren't provided:"
            f"{missing_input_variables}"
        )

    def __add__(self, other: Any) -> Template[T]:
        """
        Optionally allow combining templates
        """
        raise NotImplementedError(
            f"Template {type(self)} doesn't allow adding"
        )


class StringTemplate(Template[str]):
    input_string: str
    formatter: Formatter

    def __init__(
        self,
        input_string: str,
        input_variables: Optional[List[str]] = None,
        formatter: Formatter = Formatter(),
    ):
        super().__init__(input_variables)
        self.input_string = input_string
        self.formatter = formatter

    @classmethod
    def from_input(cls, input_string: str) -> Template[str]:
        return StringTemplate(input_string)

    @classmethod
    def from_file(cls, file_to_read: file) -> Template[str]:
        string_data = file_to_read.read()

        return StringTemplate(string_data)

    def to_file(self, file_name: str):
        file_to_write = open(file_name, "w")
        file_to_write.write(self.input_string)
        file_to_write.close()

    def _format(self, **kwargs: Any) -> str:
        if len(kwargs) == 0:
            return self.input_string
        return self.formatter.format(self.input_string, **kwargs)

    def __add__(self, other: Any) -> Template[str]:
        if isinstance(other, StringTemplate):
            return StringTemplate(
                self.input_string + other.input_string,
                self.input_variables + other.input_variables,
                self.formatter,
            )
        if isinstance(other, str):
            return StringTemplate(
                self.input_string + other,
                self.input_variables.copy(),
                self.formatter,
            )
        raise NotImplementedError(
            "StringTemplate only allows addition"
            "with another StringTemplate or str"
        )


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

    @classmethod
    def from_file(cls, file_to_read: file) -> Template[List[dict]]:
        return ChatTemplate(pickle.load(file_to_read))

    def to_file(self, file_name: str):
        with open(file_name, "wb") as f:
            pickle.dump(self.input_list, f)

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

class MaskToSentinelTemplate(StringTemplate):
    def __init__(
        self,
        input_string: str,
        input_variables: Optional[List[str]] = None,
        formatter: Formatter = Formatter(),
        mask_token: str = "<mask>",
        sentinel_token_template: str = "<extra_id_{}>",
        sentinel_start_index = 0,
        sentinel_end_index = 99,
    ):
        super().__init__(input_string, input_variables, formatter)
        self.mask_token = mask_token
        self.sentinel_token_template = sentinel_token_template
        self.sentinel_start_index = sentinel_start_index
        self.sentinel_end_index = sentinel_end_index

    @classmethod
    def from_input(cls, input_string: str) -> Template[str]:
        return MaskToSentinelTemplate(input_string)

    @classmethod
    def from_file(cls, file_to_read: file) -> Template[str]:
        string_data = file_to_read.read()

        return MaskToSentinelTemplate(string_data)

    def to_file(self, file_name: str):
        file_to_write = open(file_name, "w")
        file_to_write.write(self.input_string)
        file_to_write.close()

    def _apply_sentinel_tokens(self, prompt: str) -> str:
        split_prompt = prompt.split(self.mask_token)

        response = ""
        for index, prompt_part in enumerate(split_prompt):
            if index > 0:
                response += self.sentinel_token_template.format((index-1)%(self.sentinel_end_index-self.sentinel_start_index+1)+self.sentinel_start_index)
            response += prompt_part
        
        return response

    def _format(self, **kwargs: Any) -> str:
        if len(kwargs) == 0:
            return self._apply_sentinel_tokens(self.input_string)
        return self._apply_sentinel_tokens(self.formatter.format(self.input_string, **kwargs))

    def __add__(self, other: Any) -> Template[str]:
        if isinstance(other, StringTemplate):
            return StringTemplate(
                self.input_string + other.input_string,
                self.input_variables + other.input_variables,
                self.formatter,
            )
        if isinstance(other, str):
            return StringTemplate(
                self.input_string + other,
                self.input_variables.copy(),
                self.formatter,
            )
        raise NotImplementedError(
            "StringTemplate only allows addition"
            "with another StringTemplate or str"
        )
