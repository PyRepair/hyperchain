from __future__ import annotations
from abc import ABC, abstractmethod
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
    def from_file(cls, file_to_read: file) -> Template[T]:
        """
        Load template from file
        """
        return pickle.load(file_to_read)


    def to_file(self, file_name: str):
        """
        Save template to file
        """
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

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