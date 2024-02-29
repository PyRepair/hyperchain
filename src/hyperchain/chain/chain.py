from __future__ import annotations
from typing import List, Any, Dict, Optional
from abc import ABC, abstractmethod

import asyncio

from .chain_result import ChainResult
from ..prompt_templates import Template

class Chain(ABC):
    required_keys: Optional[List[str]] = None
    output_keys: Optional[List[str]] = None

    def run(self, **inputs_dict: Any) -> ChainResult:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_run(**inputs_dict))

    @abstractmethod
    async def async_run(self, **inputs_dict: Any) -> ChainResult:
        raise NotImplementedError()

    def run_multiple(self, *inputs_dict: Dict[str, Any]) -> List[ChainResult]:
        return asyncio.get_event_loop().run_until_complete(
            self.async_run_multiple(*inputs_dict)
        )

    async def async_run_multiple(
        self, *inputs_dict: Dict[str, Any]
    ) -> List[ChainResult]:
        return await asyncio.gather(
            *[self.async_run(**input_list) for input_list in inputs_dict]
        )

    def __add__(self, other: Any) -> Chain:
        """
        Optionally allow combining chains
        """
        raise NotImplementedError(
            f'Chain type "{type(self)}" doesn\'t implement addition'
        )