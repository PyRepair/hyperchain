from __future__ import annotations
from typing import List, Any, Dict
from dataclasses import dataclass

import asyncio

from .chain_result import ChainResult, ChainResultList
from .function_chain import FunctionChain
from .chain import Chain

@dataclass
class PriorityValue:
    value: Any
    priority: int = -1

    def increase_priority(self, amount: int):
        self.priority += amount

class PriorityDict:
    def __init__(self, items: Dict[Any, Any] = None, priority: int = -1):
        self.dict = {k: PriorityValue(v, priority) for k, v in items.items()} if items else {}

    def update_with_higher_priority(self, other: PriorityDict):
        for key, value in other.dict.items():
            if key not in self.dict or self.dict[key].priority < value.priority:
                self.dict[key] = value

    def unwrap(self) -> Dict[Any, Any]:
        return {key: value.value for key, value in self.dict.items()}

    def increase_priority_of_keys(self, keys: List[Any], amount: int):
        for key in keys:
            if key in self.dict:
                self.dict[key].increase_priority(amount)

class ChainSequenceElement:

    def __init__(self, dependency_count, chain_runner, priority):
        self.lock = asyncio.Lock()
        self.dependency_count = dependency_count
        self.chain_runner = chain_runner
        self.priority = priority
        self.input_dict = PriorityDict()
        self.result = None
        self.listeners = []
    
    def add_listener(self, listener):
        self.listeners.append(listener)

    async def notify_done(self, input_dict):
        async with self.lock:
            self.dependency_count -= 1
            self.input_dict.update_with_higher_priority(input_dict)

            if self.dependency_count == 0:
                self.result = PriorityDict((await self.chain_runner.async_run(**self.input_dict.unwrap())).output_dict, self.priority)
                if self.chain_runner.output_keys is not None:
                    self.result.increase_priority_of_keys(self.chain_runner.output_keys, 1e8)
                else:
                    self.result.increase_priority_of_keys(self.result.dict.keys(), 1e8)
                
                for listener in self.listeners:
                    await listener.notify_done(self.result)

class ChainSequenceMultiElement:
    def __init__(self, dependency_count, chain_runner, priority, dict_count):
        self.lock = asyncio.Lock()
        self.dependency_count = dependency_count
        self.chain_runner = chain_runner
        self.priority = priority
        self.input_list = [PriorityDict() for _ in range(dict_count)]
        self.result = None
        self.listeners = []
    
    def add_listener(self, listener):
        self.listeners.append(listener)

    async def notify_done(self, input_list):
        async with self.lock:
            self.dependency_count -= 1
            for input_dict, input_dict_new in zip(self.input_list, input_list):
                input_dict.update_with_higher_priority(input_dict_new)

            if self.dependency_count == 0:
                self.result = [PriorityDict(output_dict, priority=self.priority) for output_dict in (await self.chain_runner.async_run_multiple(*[input_dict.unwrap() for input_dict in self.input_list])).output_dict]
                for result in self.result:
                    if self.chain_runner.output_keys is not None:
                        result.increase_priority_of_keys(self.chain_runner.output_keys, 1e8)
                    else:
                        result.increase_priority_of_keys(result.dict.keys(), 1e8)
                
                for listener in self.listeners:
                    await listener.notify_done(self.result)

class ChainSequence(Chain):
    chains: List[Chain]

    def __init__(self, chains: List[Chain]):
        self.chains = chains

    async def async_run(self, **inputs_dict: Any) -> ChainResult:
        sequence_elements = chains_to_sequence_elements(self.chains)
        
        inputs_dict_with_priority = PriorityDict(inputs_dict, -1)

        await asyncio.gather(*[element.notify_done(inputs_dict_with_priority) for element in sequence_elements])

        previous_result = None
        priority_dict = PriorityDict(inputs_dict, -1)
        for element in sequence_elements:
            priority_dict.update_with_higher_priority(element.result)
            result = ChainResult(priority_dict.unwrap())
            
            if previous_result is not None:
                result.previous_result = previous_result
                previous_result.next_result = result
            
            previous_result = result

        return previous_result
    
    async def async_run_multiple(self, *inputs_list: Any) -> ChainResult:
        if len(inputs_list) == 1:
            return ChainResultList([await self.async_run(**(inputs_list[0]))])
        
        sequence_elements = chains_to_sequence_elements(self.chains, len(inputs_list))
        
        inputs_list_with_priority = [PriorityDict(inputs_dict, -1) for inputs_dict in inputs_list]

        await asyncio.gather(*[element.notify_done(inputs_list_with_priority) for element in sequence_elements])

        previous_result = [None for _ in range(len(inputs_list))]
        priority_dict_list = [PriorityDict(inputs_dict, -1) for inputs_dict in inputs_list]
        for element in sequence_elements:
            for i, (priority_dict, element_result) in enumerate(zip(priority_dict_list, element.result)):
                priority_dict.update_with_higher_priority(element_result)
                result = ChainResult(priority_dict.unwrap())
                
                if previous_result[i] is not None:
                    result.previous_result = previous_result[i]
                    previous_result[i].next_result = result
                
                previous_result[i] = result

        return ChainResultList(previous_result)

    def __add__(self, other) -> Chain:
        if isinstance(other, ChainSequence):
            return ChainSequence(self.chains + other.chains)

        if (not isinstance(other, Chain)) and callable(other):
            return ChainSequence(self.chains + [FunctionChain(other)])

        return ChainSequence(self.chains + [other])

def chains_to_sequence_elements(chains, dict_count = 1):
    dependency_dict = {}
    unspecified_outputs = set()
    sequence_elements = []
    for index, chain in enumerate(chains):
        dependency_list = list(range(index))
        if chain.required_keys is not None:
            dependency_list = list({dependency_dict[key] for key in chain.required_keys if key in dependency_dict} | unspecified_outputs)
        
        if dict_count > 1:
            element = ChainSequenceMultiElement(len(dependency_list) + 1, chain, index, dict_count)
        else:
            element = ChainSequenceElement(len(dependency_list) + 1, chain, index)

        for dependency_index in dependency_list:
            sequence_elements[dependency_index].add_listener(element)

        sequence_elements.append(element)
        
        if chain.output_keys is None:
            unspecified_outputs.add(index)
        else:
            for key in chain.output_keys:
                dependency_dict[key] = index
    
    return sequence_elements