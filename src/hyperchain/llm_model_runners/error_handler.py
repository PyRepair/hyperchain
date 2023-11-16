from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

from .llm_result import LLMResult


class ErrorHandlerResponse(ABC):
    pass


@dataclass
class ThrowExceptionResponse(ErrorHandlerResponse):
    exception: Any


@dataclass
class WaitResponse(ErrorHandlerResponse):
    delay: float = 0.0


class BaseErrorHandler:
    handled_error_types: List[Any] = []

    def on_run(self):
        pass

    @abstractmethod
    def on_error(self, exception: Any) -> ErrorHandlerResponse:
        raise NotImplementedError()

    def on_success(self, result: LLMResult):
        pass
