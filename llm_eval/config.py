"""Configuration for the eval toolkit."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    eval_model: str = "claude-sonnet-4-5-20250929"
    eval_concurrency: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
