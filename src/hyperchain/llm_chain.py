from typing import List, Any, Dict, Optional
from abc import ABC, abstractmethod
import logging
from .prompt_templates import Template
from .llm_model_runners.llm_result import LLMResult
from .llm_model_runners.llm_runner import LLMRunner
from .llm_model_runners.error_handler import (
    BaseErrorHandler,
    ErrorHandlerResponse,
    WaitResponse,
    ThrowExceptionResponse,
)
import asyncio
from time import sleep, time


class Chain(ABC):
    output_name: str = "result"

    def run(self, **inputs_list: Any) -> LLMResult:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_run(**inputs_list))

    @abstractmethod
    async def async_run(self, **inputs_list: Any) -> LLMResult:
        raise NotImplementedError()

    def run_multiple(self, *inputs_list: Dict[str, Any]) -> List[LLMResult]:
        return asyncio.get_event_loop().run_until_complete(
            self.async_run_multiple(*inputs_list)
        )

    async def async_run_multiple(
        self, *inputs_list: Dict[str, Any]
    ) -> List[LLMResult]:
        return await asyncio.gather(
            *[self.async_run(**input_list) for input_list in inputs_list]
        )

    def __add__(self, other: Any) -> "Chain":
        """
        Optionally allow combining chains
        """
        raise NotImplementedError(
            f'Chain type "{type(self)}" doesn\'t implement addition'
        )


class LLMChain(Chain):
    template: Template
    llm_runner: LLMRunner

    _error_handling_lock = asyncio.Lock()

    _rate_limited_state = False
    _wait_until = 0

    def __init__(
        self,
        template: Template,
        llm_runner: LLMRunner,
        output_name: str = "result",
    ):
        self.template = template
        self.llm_runner = llm_runner
        self.output_name = output_name

    async def async_run(self, **inputs_list: Any) -> List[str]:
        handlers = self.llm_runner._get_error_handlers()
        prompt = self.template.format(**inputs_list)
        while True:
            holds_a_lock = False
            try:
                await self._error_handling_lock.acquire()
                holds_a_lock = True
                
                c_time = time()
                if self._wait_until > c_time:
                    holds_a_lock = False
                    self._error_handling_lock.release()
                    await asyncio.sleep(self._wait_until - c_time)
                    continue
                

                try:
                    for handler in handlers:
                        handler.on_run()

                    if not self._rate_limited_state:
                        holds_a_lock = False
                        self._error_handling_lock.release()

                    result = await self.llm_runner.async_run(prompt)

                    if not holds_a_lock:
                        async with self._error_handling_lock:
                            for handler in handlers:
                                handler.on_success(result)
                    else:
                        for handler in handlers:
                            handler.on_success(result)

                    if holds_a_lock:
                        holds_a_lock = False
                        self._rate_limited_state = False
                        self._error_handling_lock.release()
                    return result

                finally:
                    if holds_a_lock:
                        holds_a_lock = False
                        self._error_handling_lock.release()

            except BaseException as exception:
                c_time = time()
                if not holds_a_lock:
                    await self._error_handling_lock.acquire()
                    holds_a_lock = True

                error_handler = None
                for handler in handlers:
                    if any(
                        isinstance(exception, handled_exception)
                        for handled_exception in handler.handled_error_types
                    ):
                        error_handler = handler
                        break

                if error_handler is None:
                    raise exception

                error_response = error_handler.on_error(exception)

                if isinstance(error_response, ThrowExceptionResponse):
                    raise error_response.exception

                if isinstance(error_response, WaitResponse):
                    if self._wait_until < c_time + error_response.delay:
                        logging.warning(
                            f"Got exception: {exception.__class__.__name__}\n"
                            f"Retrying in {error_response.delay:.2f}s... "
                        )
                        self._wait_until = c_time + error_response.delay
                    continue

                logging.error(
                    "Unknown error handling response"
                    f"{error_response.__class__.__name__}."
                )
                raise
            finally:
                if holds_a_lock:
                    holds_a_lock = False
                    self._error_handling_lock.release()

    def __add__(self, other) -> Chain:
        if isinstance(other, LLMChainSequence):
            return LLMChainSequence([self] + other.chains, other.output_name)

        return LLMChainSequence([self, other], other.output_name)


class LLMChainSequence(LLMChain):
    chains: List[LLMChain]

    def __init__(self, chains: List[LLMChain], output_name: str = "result"):
        self.chains = chains
        self.output_name = output_name

    async def async_run(self, **inputs_list: Any) -> LLMResult:
        inputs_dict = dict(inputs_list)
        result = None

        for chain in self.chains:
            last_result = result
            result = await chain.async_run(**inputs_dict)
            result.previous_result = last_result
            if last_result is not None:
                last_result.next_result = result
            inputs_dict = {**inputs_dict, chain.output_name: result.output}

        if result is None:
            return LLMResult("")

        return result

    def __add__(self, other) -> Chain:
        if isinstance(other, LLMChainSequence):
            return LLMChainSequence(
                self.chains + other.chains, other.output_name
            )

        return LLMChainSequence(self.chains + [other], other.output_name)
