from abc import ABC, abstractmethod
from .error_handler import BaseErrorHandler
from .llm_result import LLMResult
from typing import List, Any
from asyncio import create_task, gather

class LLMRunner(ABC):
    @abstractmethod
    async def async_run(self, prompt: Any) -> LLMResult:
        pass

    async def run_batch(self, prompts: List[Any]) -> List[LLMResult]:
        return await gather(*[self.async_run(prompt=prompt) for prompt in prompts])

    def _get_error_handlers(self) -> List[BaseErrorHandler]:
        return []
