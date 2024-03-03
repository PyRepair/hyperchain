import pytest
from hyperchain.prompt_templates.mask_to_sentinel_template import MaskToSentinelTemplate

@pytest.mark.parametrize("input_string, format_params, expected_string", [
    ("test <mask> <mask>", {}, "test <extra_id_0> <extra_id_1>"),
    ("test {key} <mask>", {"key": "<mask>"}, "test <extra_id_0> <extra_id_1>"),
    ("test {key1} <mask> {key2}", {"key1": "<not_a_mask> <mask>", "key2": "test <mask> test"}, "test <not_a_mask> <extra_id_0> <extra_id_1> test <extra_id_2> test"),
])
def test_sentinel_replacement(input_string, format_params, expected_string):
    sentinel_template = MaskToSentinelTemplate(input_string=input_string)
    assert sentinel_template.format(**format_params) == expected_string
