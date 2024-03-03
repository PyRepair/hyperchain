from typing import List, Any, Callable, Optional, Tuple
from copy import deepcopy

import inspect
import asyncio

from .chain_result import ChainResult
from .chain import Chain


class FunctionMapChain(Chain):
    function_map: List[Tuple[str, Callable, Optional[str]]]
    
    def __init__(
        self,
        function_map: List[Tuple[str, Callable, Optional[str]]],
    ):
        self.function_map = function_map
        self.required_keys = []
        self.output_keys = []
        for function_tuple in self.function_map:
            input_name, _, output_name = (function_tuple + function_tuple[:1])[:3]
            self.required_keys.append(input_name)
            if output_name is not None:
                self.output_keys.append(output_name)

    async def async_run(self, **inputs_dict: Any):
        output_dict = deepcopy(inputs_dict)
        for function_tuple in self.function_map:
            input_name, function, output_name = (function_tuple + function_tuple[:1])[:3]
            if input_name in inputs_dict:
                result = function(inputs_dict[input_name])
                if inspect.isawaitable(result):
                    result = await result
                output_dict[output_name] = result
        return ChainResult(output_dict=output_dict)

    def __add__(self, other) -> Chain:
        from .chain_sequence import ChainSequence
        return ChainSequence([self]) + other