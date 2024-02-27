from typing import List, Any, Callable

import inspect
import asyncio

from .chain_result import ChainResult
from .chain import Chain


class FunctionChain(Chain):
    function: Callable
    
    def __init__(
        self,
        function: Callable,
    ):
        self.function = function
    
    def run(self, **inputs_dict: Any):
        result = self.function(inputs_dict)
        if inspect.isawaitable(result):
            loop = asyncio.get_event_loop()
            return ChainResult(output_dict=loop.run_until_complete(result))
        return ChainResult(output_dict=result)

    async def async_run(self, **inputs_dict: Any):
        result = self.function(inputs_dict)
        if inspect.isawaitable(result):
            result = await result
        return ChainResult(output_dict=result)

    def __add__(self, other) -> Chain:
        from .chain_sequence import ChainSequence
        return ChainSequence([self]) + other