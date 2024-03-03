import pytest
from requests.exceptions import HTTPError, ConnectionError
from unittest.mock import Mock, patch
from hyperchain.llm_runners.hf_runner import HuggingFaceRunner, HuggingFaceHttpErrorHandler
from hyperchain.llm_runners.error_handler import WaitResponse, ThrowExceptionResponse

@pytest.mark.asyncio
@patch('requests.post')
async def test_hugging_face_runner_async_run(request_mock):
        request_mock.return_value.json = Mock(return_value={"generated_text": "test response"})
        request_mock.return_value.raise_for_status = Mock()
        
        runner = HuggingFaceRunner(api_key="dummy_key", model="dummy_model")
        result = await runner.async_run("test prompt")

        request_mock.assert_called_once_with(
            "https://api-inference.huggingface.co/models/dummy_model",
            headers={"Authorization": f"Bearer dummy_key"},
            json="test prompt"
        )
        assert str(result) == "test response"

@pytest.mark.parametrize("exception", [
    HTTPError(),
    ConnectionError(),
])
def test_hugging_face_http_error_handler(exception):
    handler = HuggingFaceHttpErrorHandler()

    for _ in range(handler._max_attempts):
        handler.on_run()
        response = handler.on_error(exception)
        assert response == WaitResponse(pow(handler._base, handler._current_attempt))

    handler.on_run()
    final_response = handler.on_error(exception)
    assert final_response == ThrowExceptionResponse(exception=exception)

    handler.on_success(result=Mock())
    assert handler._current_attempt == 0
