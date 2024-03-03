import os
import pytest
from unittest.mock import patch
from hyperchain.llm_runners.utils import get_api_key_from_env

@pytest.mark.parametrize("environ_dict, env_key, expected_result", [
    ({"ENV_KEY": "secret_key"}, "ENV_KEY", "secret_key"),
    ({"ENV_KEY": "secret_key"}, "WRONG_ENV_KEY", ""),
    ({}, "ENV_KEY", ""),
])
def test_get_api_key_from_env(environ_dict, env_key, expected_result):
    with patch.dict(os.environ, environ_dict):
        assert get_api_key_from_env(env_key) == expected_result
