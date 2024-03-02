import pytest
from hyperchain.prompt_templates.string_template import StringTemplate
from unittest.mock import MagicMock, mock_open, patch

@pytest.mark.parametrize("input_string, required_keys", [
    ("no keys", []),
    ("there's a {key}", ["key"]),
    ("{multiple} keys {used}", ["multiple", "used"]),
    ("escaped {{brackets}}", []),
])
def test_required_keys_detection(input_string, required_keys):
    assert StringTemplate(input_string=input_string).required_keys == required_keys

@patch("builtins.open", new_callable=mock_open)
def test_saving_to_file(mock_file):
    template = StringTemplate(input_string="testing!")
    template.to_file("file.txt")

    mock_file.assert_called_once_with("file.txt", "w")
    mock_file().write.assert_called_once_with("testing!")

@patch("builtins.open", new_callable=mock_open, read_data="testing!")
def test_loading_from_file(mock_file):
    template = StringTemplate.from_file("file.txt")
    mock_file.assert_called_once_with("file.txt", "r")

    assert template.format() == "testing!"