from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar, Any, Optional
import pickle

T = TypeVar("T")


class Template(Generic[T], ABC):
    required_keys: Optional[List[str]] = None

    @classmethod
    @abstractmethod
    def from_input(cls, input: T) -> Template[T]:
        """
        Load template directly from specified input type
        """

    @classmethod
    def from_file(cls, file_to_read: str) -> Template[T]:
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
    def format(self, **kwargs: Any) -> str:
        """
        Format string to be passed to the model
        """

    def __add__(self, other: Any) -> Template[T]:
        """
        Optionally allow combining templates
        """
        raise NotImplementedError(
            f"Template {type(self)} doesn't allow adding"
        )