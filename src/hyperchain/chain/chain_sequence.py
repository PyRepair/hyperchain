from typing import List, Any, Dict

from .chain_result import ChainResult
from .function_chain import FunctionChain
from .chain import Chain

class ChainSequence(Chain):
    chains: List[Chain]

    def __init__(self, chains: List[Chain]):
        self.chains = chains

    async def async_run(self, **inputs_dict: Any) -> ChainResult:
        result = None

        for chain in self.chains:
            last_result = result
            if result is None:
                result = await chain.async_run(**inputs_dict)
            else:
                result = await chain.async_run(**result.output_dict)
                result.previous_result = last_result
                last_result.next_result = result

        return result

    def __add__(self, other) -> Chain:
        if isinstance(other, ChainSequence):
            return ChainSequence(self.chains + other.chains)

        if (not isinstance(other, Chain)) and callable(other):
            return ChainSequence(self.chains + [FunctionChain(other)])

        return ChainSequence(self.chains + [other])
