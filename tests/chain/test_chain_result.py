import pytest
from hyperchain.chain.chain_result import ChainResult

@pytest.mark.parametrize("output_dict, attribute_name, expected_value", [
    ({"a": "b"}, "a", "b"),
    ({"output_dict": "b"}, "output_dict", {"output_dict": "b"}),
    ({"previous_result": "b"}, "previous_result", None),
    ({"next_result": "b"}, "next_result", None),
    ({}, "a", None),
])
def test_attribute_retireval(output_dict, attribute_name, expected_value):
    result = ChainResult(output_dict=output_dict)
    assert result.__getattr__(attribute_name) == expected_value
