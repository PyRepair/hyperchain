import os

def get_api_key_from_env(api_key: str) -> str:
    return (
        os.environ[api_key]
        if api_key in os.environ and os.environ[api_key]
        else ""
    )