import os
from typing import Optional

def get_litellm_api_key() -> str:
    """Get LiteLLM API key from environment variable."""
    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        raise ValueError(
            "LITELLM_API_KEY environment variable not set. "
            "Please set it in your ~/.bashrc or environment."
        )
    return api_key

def get_litellm_base_url() -> str:
    """Get LiteLLM base URL from environment variable."""
    base_url = os.getenv("LITELLM_BASE_URL", "https://cmu.litellm.ai")
    return base_url

def get_litellm_config() -> tuple[str, str]:
    """Get both API key and base URL."""
    return get_litellm_api_key(), get_litellm_base_url() 