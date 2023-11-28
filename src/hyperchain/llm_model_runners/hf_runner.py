from typing import Optional
from .llm_runner import LLMRunner, LLMResult
import requests
import os

from .error_handler import (
    BaseErrorHandler,
    ThrowExceptionResponse,
    WaitResponse,
    ErrorHandlerResponse,
)

HF_ENV_KEY = "HUGGINGFACEHUB_API_TOKEN"


def _get_api_key_from_env() -> str:
    return (
        os.environ[HF_ENV_KEY]
        if HF_ENV_KEY in os.environ and os.environ[HF_ENV_KEY]
        else ""
    )


class HuggingFaceRunner(LLMRunner):
    api_key: str
    model_args: Optional[dict]

    def __init__(
        self,
        model: str = "gpt2",
        json_result_name: str = "generated_text",
        api_key: str = _get_api_key_from_env(),
        model_args: Optional[dict] = None,
    ):
        self.model = model
        self.json_result_name = json_result_name
        self.model_args = model_args
        self.api_key = api_key
        self._error_handlers = [HuggingFaceHttpErrorHandler(api_key)]

    async def async_run(self, prompt: str):
        if self.api_key is not None and len(self.api_key) > 0:
            headers = {"Authorization": f"Bearer {self.api_key}"}
        else:
            headers = {}
        API_URL = f"https://api-inference.huggingface.co/models/{self.model}"

        response = requests.post(API_URL, headers=headers, json=prompt)
        response.raise_for_status()

        response_json = response.json()
        if isinstance(response_json, list) and len(response_json) == 1:
            response_json = response_json[0]

        return LLMResult(response_json[self.json_result_name], response.json())

    def _get_error_handlers(self):
        return self._error_handlers


class HuggingFaceHttpErrorHandler(BaseErrorHandler):
    handled_error_types = [
        requests.exceptions.HTTPError,
        requests.exceptions.ConnectionError,
    ]

    _current_attempt = 0
    _max_attempts = 5
    _base = 1.5

    _requests_sent = 0
    _responses_received = 0
    _increase_since_request = 0

    def __init__(self, api_key: str):
        self.api_key = api_key

    def on_run(self):
        self._requests_sent += 1

    def on_error(self, exception):
        self._responses_received += 1
        if self._current_attempt < self._max_attempts:
            if self._responses_received >= self._increase_since_request:
                self._current_attempt += 1
                self._increase_since_request = self._requests_sent + 1

            return WaitResponse(delay=pow(self._base, self._current_attempt))

        return ThrowExceptionResponse(exception)

    def on_success(self, result):
        self._responses_received += 1
        self._current_attempt = 0
