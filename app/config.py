"""Configuration for FastAPI GitHub App."""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # FastAPI settings
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )

    # GitHub App settings
    github_app_id: str = Field(..., env="GITHUB_APP_ID")
    github_app_private_key: str = Field(..., env="GITHUB_APP_PRIVATE_KEY")
    github_webhook_secret: str = Field(..., env="GITHUB_WEBHOOK_SECRET")

    # LLM settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    yandex_gpt_api_key: Optional[str] = Field(default=None, env="YANDEX_GPT_API_KEY")
    yandex_gpt_folder_id: Optional[str] = Field(default=None, env="YANDEX_GPT_FOLDER_ID")

    # Agent settings
    max_iterations: int = Field(default=5, env="MAX_ITERATIONS")
    code_agent_name: str = Field(default="AI Code Agent", env="CODE_AGENT_NAME")
    reviewer_agent_name: str = Field(default="AI Reviewer Agent", env="REVIEWER_AGENT_NAME")

    # Database settings
    database_url: str = Field(default="sqlite:///./app.db", env="DATABASE_URL")

    # Service settings
    webhook_timeout: int = Field(default=30, env="WEBHOOK_TIMEOUT")
    ci_check_interval: int = Field(default=60, env="CI_CHECK_INTERVAL")  # seconds
    ci_max_wait_time: int = Field(default=1800, env="CI_MAX_WAIT_TIME")  # 30 minutes

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def use_yandex_gpt(self) -> bool:
        """Check if Yandex GPT should be used instead of OpenAI."""
        return bool(self.yandex_gpt_api_key and self.yandex_gpt_folder_id)

    def get_private_key(self) -> str:
        """Get GitHub App private key content."""
        # If it's a file path, read the file
        if self.github_app_private_key.startswith("/") or self.github_app_private_key.endswith(".pem"):
            try:
                with open(self.github_app_private_key, "r") as f:
                    return f.read()
            except FileNotFoundError:
                raise ValueError(f"Private key file not found: {self.github_app_private_key}")
        
        # Otherwise, treat as direct key content
        return self.github_app_private_key


# Global settings instance
settings = Settings()