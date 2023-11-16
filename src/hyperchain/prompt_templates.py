from __future__ import annotations
from abc import ABC, abstractmethod
from string import Formatter
from typing import List, Generic, TypeVar, Any, Optional

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
    def from_file(cls, file_name: str) -> Template[T]:
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
    def from_file(cls, file_name: str) -> Template[str]:
        file_to_read = open(file_name, "r")
        string_data = file_to_read.read()
        file_to_read.close()

        return StringTemplate(string_data)

    def to_file(self, file_name: str):
        file_to_write = open(file_name, "w")
        file_to_write.write(self.input_string)
        file_to_write.close()

    def _format(self, **kwargs: Any) -> str:
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
