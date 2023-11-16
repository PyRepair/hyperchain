from abc import ABC, abstractmethod
from .error_handler import BaseErrorHandler
from .llm_result import LLMResult
from typing import List


class LLMRunner(ABC):
    @abstractmethod
    async def async_run(self, prompt: str) -> LLMResult:
        pass

    def _get_error_handlers(self) -> List[BaseErrorHandler]:
        return []
