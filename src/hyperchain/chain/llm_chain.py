from typing import List, Any
import logging

from .chain_result import ChainResult
from .chain import Chain
from .chain_sequence import ChainSequence

from ..prompt_templates import Template
from ..llm_runners.llm_runner import LLMRunner
from ..llm_runners.error_handler import (
    WaitResponse,
    ThrowExceptionResponse,
)
import asyncio
from time import time

class LLMChain(Chain):
    template: Template
    llm_runner: LLMRunner
    output_name: str

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

    async def async_run(self, **inputs_dict: Any) -> ChainResult:
        handlers = self.llm_runner._get_error_handlers()
        prompt = self.template.format(**inputs_dict)
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
                    output_dict = inputs_dict
                    output_dict[self.output_name] = result
                    return ChainResult(output_dict=output_dict)

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
        return ChainSequence([self]) + other