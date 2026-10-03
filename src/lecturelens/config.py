"""Application settings, loaded once from environment variables / the .env file.

Using pydantic-settings means every setting is typed and validated at startup:
a missing API key fails immediately with a clear error instead of crashing
halfway through a request.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # SecretStr hides the value in logs and error messages (prints as '**********').
    google_api_key: SecretStr

    chat_model: str = "gemini-3.8-flash"
    embedding_model: str = "gemini-embedding-2"

    # Where cached transcripts and the vector database are stored (git-ignored).
    data_dir: Path = Path("data")

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, reading .env only on the first call."""
    return Settings()
