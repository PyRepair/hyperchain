import pytest
from unittest.mock import Mock
from hyperchain.llm_runners.t5_model_runner import T5ConditionalModelRunner

def test_apply_response():
    mocked_model = Mock()
    mocked_tokenizer = Mock()

    mocked_tokenizer.additional_special_tokens = ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>"]
    mocked_tokenizer.convert_tokens_to_ids.return_value = [1, 2, 3]

    model_runner = T5ConditionalModelRunner(mocked_model, tokenizer=mocked_tokenizer)
    mocked_tokenizer.convert_tokens_to_ids.assert_called_once_with(["<extra_id_0>", "<extra_id_1>", "<extra_id_2>"])

    mocked_tokenizer.get_special_tokens_mask.return_value = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]

    result = model_runner._apply_response(Mock(tolist=lambda: [[1, 2, 3]]), Mock(tolist=lambda: [[1, 4, 5, 6, 2, 7, 8, 3, 9, 10, 11]]))
    assert result == [[4, 5, 6, 7, 8, 9, 10]]

@pytest.mark.asyncio
async def test_async_run():
    mocked_model = Mock()
    mocked_tokenizer = Mock()

    mocked_tokenizer.additional_special_tokens = ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>"]
    mocked_tokenizer.convert_tokens_to_ids.return_value = [1, 2, 3]

    model_runner = T5ConditionalModelRunner(mocked_model, tokenizer=mocked_tokenizer)
    mocked_tokenizer.convert_tokens_to_ids.assert_called_once_with(["<extra_id_0>", "<extra_id_1>", "<extra_id_2>"])

    mocked_tokenizer.get_special_tokens_mask.return_value = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    mocked_tokenizer.return_value = Mock(input_ids=Mock(tolist=lambda: [[1, 2, 3]]))
    mocked_tokenizer.decode.return_value = "dummy response"

    mocked_model.generate.return_value = Mock(tolist=lambda: [[1, 4, 5, 6, 2, 7, 8, 3, 9, 10, 11]])

    result = await model_runner.async_run("dummy prompt")

    mocked_model.generate.assert_called_once_with(mocked_tokenizer.return_value.input_ids)

    assert result.output == "dummy response"