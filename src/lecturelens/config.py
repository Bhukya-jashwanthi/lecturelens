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

    # A "lite" model: answering from a few retrieved passages doesn't need the largest
    # model, and it responds in about a second. gemini-3.8-flash's free tier allows only
    # 20 requests per day. Grounding was checked with eval/evaluate.py --answers.
    chat_model: str = "gemini-3.5-flash-lite"
    # Gemini 3 "thinks" before answering; little of that is needed here.
    # Accepted values depend on the model (gemini-3.8-flash rejects "minimal").
    reasoning_effort: str = "low"
    # How many retrieved chunks are given to the LLM as sources.
    top_k: int = 5
    embedding_model: str = "gemini-embedding-2"
    # Gemini embeddings can be truncated (Matryoshka): 768 keeps nearly all the
    # retrieval quality of the full 3072 at a quarter of the storage.
    embedding_dimensions: int = 768

    # Where cached transcripts and the vector database are stored (git-ignored).
    data_dir: Path = Path("data")

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, reading .env only on the first call."""
    return Settings()
