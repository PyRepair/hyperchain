from typing import Optional, List, Final, Any
from .llm_runner import LLMRunner, LLMChatRunner, LLMResult
from openai._exceptions import AuthenticationError
from openai import AsyncOpenAI
from dataclasses import dataclass
from .error_handler import (
    BaseErrorHandler,
    ThrowExceptionResponse,
    ErrorHandlerResponse,
)

import logging
import tiktoken
import asyncio
import os
import re
import time

OPENAI_ENV_KEY = "OPENAI_API_TOKEN"


def _get_api_key_from_env() -> str:
    return (
        os.environ[OPENAI_ENV_KEY]
        if OPENAI_ENV_KEY in os.environ and os.environ[OPENAI_ENV_KEY]
        else ""
    )


UNIT_TO_SECONDS_DICT = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}


def _get_seconds_from_unit_string(unit_string: str) -> float:
    total_seconds = 0.0
    for value, unit in re.findall("([0-9]+)([a-zA-Z]+)", unit_string):
        total_seconds += float(value) * UNIT_TO_SECONDS_DICT.get(unit, 0.0)
    return total_seconds


class OpenAIRequest:
    """
    Class wrapping an OpenAI API request which can be used by
    the async workers to send them easily when needed.
    It dispatches self.done_event when it either
    successfully finished or recieves an exception.

    If exception is not None after getting done_event,
    the class calling run() should handle the error.
    This object should only be run once.
    """

    result_raw = None
    output = None
    exception = None
    max_tokens = 500

    def __init__(self, api_key, model, prompt, model_params):
        self.api_key = api_key
        self.model = model
        self.prompt = prompt
        self.model_params = model_params
        if "max_tokens" in self.model_params:
            self.max_tokens = model_params.get("max_tokens")
            self.model_params.pop("max_tokens")
        self.done_event = asyncio.Event()

    @property
    def encoding(self):
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    @property
    def guessed_token_count(self):
        return self.max_tokens + len(self.encoding.encode(self.prompt))

    async def run(self, client: AsyncOpenAI):
        try:
            response = await client.completions.with_raw_response.create(
                model=self.model,
                prompt=self.prompt,
                max_tokens=self.max_tokens,
                **self.model_params,
            )
            self.result_raw = dict(response.parse())
            self.result_raw["headers"] = response.headers
            self.output = self.result_raw["choices"][0].text
        except BaseException as ex:
            self.exception = ex
        finally:
            if self.done_event:
                self.done_event.set()


class OpenAIChatRequest:
    result_raw = None
    output = None
    exception = None
    max_tokens = 500

    def __init__(self, api_key, model, chat, model_params):
        self.api_key = api_key
        self.model = model
        self.chat = chat
        self.model_params = model_params
        if "max_tokens" in self.model_params:
            self.max_tokens = model_params.get("max_tokens")
            self.model_params.pop("max_tokens")
        self.done_event = asyncio.Event()

    @property
    def encoding(self):
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    @property
    def guessed_token_count(self):
        return self.max_tokens + sum(
            len(self.encoding.encode(d["content"])) for d in self.chat
        )

    async def run(self, client: AsyncOpenAI):
        try:
            response = await client.chat.completions.with_raw_response.create(
                model=self.model,
                messages=self.chat,
                max_tokens=self.max_tokens,
                **self.model_params,
            )
            self.result_raw = dict(response.parse())
            self.result_raw["headers"] = response.headers
            self.output = self.result_raw["choices"][0].message.content
        except BaseException as ex:
            self.exception = ex
        finally:
            if self.done_event:
                self.done_event.set()


MAXIMUM_REQUEST_WAIT_TIME: Final[float] = 7.5
INITIAL_TOKEN_LIMIT: Final[int] = 20000
CONCURRENCY_LIMIT: Final[int] = 50
LIMIT_CHECK_INTERVAL: Final[tuple[int, int]] = (0.01, 0.5)


class OpenAIRequestHandler:
    """
    Class handling the asynchronous dispatching of OpenAI requests.
    It should be initialzied using get(api_key) class method, to ensure
    limits are respected per api_key and not per instance.
    """

    _instance_dict: dict = {}
    _client: AsyncOpenAI = None

    def __init__(self):
        raise RuntimeError(
            "OpenAIRequestHandler can't be initialized directly."
            + "Please use get(api_key) function instead."
        )

    @classmethod
    def get(cls, api_key: str):
        if not api_key in cls._instance_dict:
            new_instance = cls.__new__(cls)
            new_instance._client = AsyncOpenAI(api_key=api_key)
            cls._instance_dict[api_key] = new_instance

        return cls._instance_dict[api_key]

    _max_tokens: int = INITIAL_TOKEN_LIMIT

    _token_limit_queue: Final[asyncio.PriorityQueue] = asyncio.PriorityQueue()
    _token_limit_queue_lock: Final[asyncio.Lock] = asyncio.Lock()

    _queue: Final[asyncio.Queue[OpenAIRequest]] = asyncio.Queue()
    _rate_check_lock: Final[asyncio.Lock] = asyncio.Lock()

    _workers: Final[List[asyncio.Task]] = []
    _workers_lock: Final[asyncio.Lock] = asyncio.Lock()

    _total_guessed_count: int = 0
    _total_guessed_count_lock: Final[asyncio.Lock] = asyncio.Lock()

    async def _on_response_headers(self, headers):
        """
        When getting a response we look for rate limits in the headers
        We calculate a timestamp until we can resume requests and save it in
        an object variable, which we later use to rate limit our requester
        by waiting until that timestamp if we get too close to the rate limit
        """
        if "x-ratelimit-limit-tokens" in headers:
            self._max_tokens = max(
                self._max_tokens, int(headers["x-ratelimit-limit-tokens"])
            )

        if (
            "x-ratelimit-reset-tokens" in headers
            and "x-ratelimit-remaining-tokens" in headers
        ):
            token_count = int(headers["x-ratelimit-remaining-tokens"])
            time_seconds = _get_seconds_from_unit_string(
                headers["x-ratelimit-reset-tokens"]
            )
            if time_seconds > MAXIMUM_REQUEST_WAIT_TIME:
                token_count = int(
                    token_count * time_seconds / MAXIMUM_REQUEST_WAIT_TIME
                )
                time_seconds = MAXIMUM_REQUEST_WAIT_TIME
            timestamp = time.time() + time_seconds
            async with self._token_limit_queue_lock:
                self._token_limit_queue.put_nowait((token_count, timestamp))

    async def send_request(self, request: OpenAIRequest) -> LLMResult:
        """
        When sending a request, this method puts it on the queue
        for the workers to handle asynchronously and waits for a response.
        """
        await self._start_workers()
        self._queue.put_nowait(request)
        await request.done_event.wait()

        if request.exception is not None:
            raise request.exception

        async with self._workers_lock:
            if self._queue.empty():
                await self._stop_workers()

        return LLMResult(
            request.output,
            extra_llm_outputs=dict(request.result_raw),
        )

    async def _start_workers(self):
        async with self._workers_lock:
            for _ in range(max(0, CONCURRENCY_LIMIT - len(self._workers))):
                worker = asyncio.create_task(self._worker())
                self._workers.append(worker)

    async def _stop_workers(self):
        if len(self._workers) == 0:
            return

        await self._queue.join()

        for worker in self._workers:
            worker.cancel()

        self._workers.clear()

    async def _worker(self):
        """
        Method defining the main worker coroutine.
        It waits for a task and handles it, minding the possible token limits and exceptions raised.
        """
        while True:
            task = await self._queue.get()

            async with self._rate_check_lock:
                async with self._total_guessed_count_lock:
                    self._total_guessed_count += task.guessed_token_count

                current_limit = 20000
                current_tokens = 0
                wait_time = 0.0

                while True:
                    async with self._token_limit_queue_lock:
                        while (
                            len(self._token_limit_queue._queue) > 0
                            and self._token_limit_queue._queue[0][1]
                            < time.time()
                        ):
                            self._token_limit_queue.get_nowait()

                        limit_tuple = (
                            self._token_limit_queue._queue[0]
                            if len(self._token_limit_queue._queue) > 0
                            else (self._max_tokens, time.time() + 1)
                        )
                        current_limit = limit_tuple[0]
                        wait_time = min(
                            max(
                                LIMIT_CHECK_INTERVAL[0],
                                limit_tuple[1] - time.time(),
                            ),
                            LIMIT_CHECK_INTERVAL[1],
                        )

                    async with self._total_guessed_count_lock:
                        current_tokens = self._total_guessed_count

                    if current_tokens <= current_limit:
                        break

                    logging.info(
                        "Rate limit getting close "
                        f"waiting for: {wait_time}"
                        f"s, Tokens in queue {current_tokens}"
                        f" with limit {max(0, current_limit)}"
                    )
                    await asyncio.sleep(wait_time)

            try:
                await task.run(self._client)
                if task.exception is None and task.result_raw is not None:
                    await self._on_response_headers(task.result_raw["headers"])
            finally:
                self._queue.task_done()
                async with self._total_guessed_count_lock:
                    self._total_guessed_count -= task.guessed_token_count


class OpenAIRunner(LLMRunner):
    api_key: str
    result_chain: List[str] = []
    model: str
    model_params: Optional[dict]
    _request_handler: OpenAIRequestHandler = None

    def __init__(
        self,
        api_key: str = _get_api_key_from_env(),
        model: str = "gpt-3.5-turbo-instruct",
        model_params: dict = {},
    ):
        self.model_params = model_params
        self.api_key = api_key
        self.model = model
        self._request_handler = OpenAIRequestHandler.get(api_key)
        self._error_handlers = [OpenAIAutheticationErrorHandler(self.api_key)]

    async def async_run(self, prompt: str):
        request = OpenAIRequest(
            api_key=self.api_key,
            model=self.model,
            prompt=prompt,
            model_params=self.model_params,
        )
        return await self._request_handler.send_request(request)

    def _get_error_handlers(self):
        return self._error_handlers


class OpenAIChatRunner(LLMChatRunner):
    api_key: str
    result_chain: List[str] = []
    model: str
    model_params: Optional[dict]
    _request_handler: OpenAIRequestHandler = None

    def __init__(
        self,
        api_key: str = _get_api_key_from_env(),
        model: str = "gpt-3.5-turbo",
        model_params: dict = {},
    ):
        self.model_params = model_params
        self.api_key = api_key
        self.model = model
        self._request_handler = OpenAIRequestHandler.get(api_key)
        self._error_handlers = [OpenAIAutheticationErrorHandler(self.api_key)]

    async def async_run(self, messages: List[dict]):
        request = OpenAIChatRequest(
            api_key=self.api_key,
            model=self.model,
            chat=messages,
            model_params=self.model_params,
        )
        return await self._request_handler.send_request(request)

    def _get_error_handlers(self):
        return self._error_handlers


# Here we define the error handlers for the errors we may encounter
# when interacting with OpenAI's API


class OpenAIAutheticationErrorHandler(BaseErrorHandler):
    handled_error_types = [AuthenticationError]
    api_key = None

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def _censored_api_key(self):
        if len(self.api_key) < 4:
            return self.api_key

        key_prefix = "sk-" if self.api_key[:3] == "sk-" else ""
        key = self.api_key[3:] if self.api_key[:3] == "sk-" else self.api_key
        if len(key) < 6:
            return key_prefix + key

        if len(self.api_key) <= 10:
            return key_prefix + key[:2] + ((len(key) - 4) * "*") + key[-2:]

        return key_prefix + key[:4] + ((len(key) - 8) * "*") + key[-4:]

    def on_error(self, exception):
        return ThrowExceptionResponse(
            ValueError(
                f'Incorrect OpenAI api key "{self._censored_api_key}", are you'
                " sure you passed a correct one in OpenAIRunner parameters or"
                ' through an "OPENAI_API_TOKEN" env variable?'
            )
        )
