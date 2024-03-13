import pytest
from unittest.mock import Mock, MagicMock, patch
from hyperchain.llm_runners.masked_model_runner import MaskedModelRunner

@pytest.mark.asyncio
@patch(
    'hyperchain.llm_runners.masked_model_runner.pipeline',
    Mock(return_value=Mock(return_value=[[{"token_str": "response"}]]))
)
async def test_async_run():
    model_runner = MaskedModelRunner(Mock(), tokenizer=Mock())

    result = await model_runner.async_run("dummy <mask>")

    assert result.output == "dummy response"

@pytest.mark.asyncio
@patch(
    'hyperchain.llm_runners.masked_model_runner.pipeline',
    Mock(return_value=Mock(return_value=[[[{"token_str": "response"}]]]))
)
@patch.dict('sys.modules', torch=MagicMock())
async def test_run_batch():
    model_runner = MaskedModelRunner(Mock(), tokenizer=Mock())

    result = await model_runner.run_batch(["dummy <mask>"])

    assert [str(res) for res in result] == ["dummy response"]