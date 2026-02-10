from typing import Optional
import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_env_file() -> str:
    """
    Robust .env file discovery.
    """
    env_file = os.getenv("ENV_FILE", ".env")
    
    # Try to find .env file if it doesn't exist in CWD
    if not os.path.exists(env_file) and not os.path.isabs(env_file):
        # Look up 3 levels
        current = Path.cwd()
        for _ in range(3):
            candidate = current / ".env"
            if candidate.exists():
                return str(candidate)
            current = current.parent
            
    return env_file


class Settings(BaseSettings):
    # LLM Configuration
    LLM_API_KEY: str = Field(default="", description="LLM API Key")
    LLM_PROVIDER: str = Field(
        default="openai", description="LLM Provider: openai, anthropic, gemini"
    )
    LLM_MODEL: str = Field(default="gpt-4o", description="LLM Model Name")
    LLM_BASE_URL: Optional[str] = Field(default=None, description="LLM API Base URL")

    # Embedding General Configuration
    EMBEDDING_PROVIDER: str = Field(
        default="openai", description="Embedding Provider: openai, local, sentence-transformers"
    )

    # OpenAI Embedding Configuration
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small", description="OpenAI Embedding Model"
    )
    EMBEDDING_BASE_URL: Optional[str] = Field(default=None, description="Embedding API Base URL")
    EMBEDDING_API_KEY: str = Field(
        default="", description="Embedding API Key (if different from LLM)"
    )

    # Local Embedding Configuration
    EMBEDDING_MODEL_PATH: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", description="Local model path or name"
    )
    EMBEDDING_MODEL_DEVICE: str = Field(
        default="auto", description="Computation device: auto, cpu, cuda, mps"
    )

    # Vector Database Configuration
    VECTOR_DB_PROVIDER: str = Field(
        default="chroma", description="Vector Database: chroma, qdrant, milvus, pinecone"
    )
    VECTOR_DB_URL: Optional[str] = Field(default=None, description="Cloud Vector Database URL")
    VECTOR_DB_API_KEY: Optional[str] = Field(default=None, description="Cloud Vector Database API Key")
    VECTOR_DB_PATH: Optional[str] = Field(default=None, description="Local Vector Database Storage Path")
    VECTOR_DB_COLLECTION: str = Field(default="conversations", description="Vector Database Collection Name")

    # Search Tool Configuration
    TAVILY_API_KEY: Optional[str] = Field(default=None, description="Tavily Search API Key")

    model_config = SettingsConfigDict(
        env_file=_get_env_file(), 
        env_file_encoding="utf-8", 
        extra="ignore", 
        case_sensitive=False,
    )


# Export configuration as singleton
settings = Settings()  # type: ignore[call-arg]
