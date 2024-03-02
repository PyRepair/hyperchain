import pytest
from hyperchain.prompt_templates.chat_template import ChatTemplate
from unittest.mock import MagicMock, mock_open, patch

@pytest.mark.parametrize("input_list, required_keys", [
    ([{"content": "No keys here"}], []),
    ([{"content": "{key} in some messages"},{"content": "but not all"}], ["key"]),
    ([{"content": "{multiple}"}, {"content": "keys"}, {"content": "{used}"}], ["multiple", "used"]),
    ([{"unrelevant": "{multiple}"}, {"unrelevant": "keys"}, {"unrelevant": "{unused}"}], []),
])
def test_required_keys_detection(input_list, required_keys):
    assert ChatTemplate(input_list=input_list).required_keys == required_keys
