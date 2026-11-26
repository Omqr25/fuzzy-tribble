from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Syriatel Matching Service"
    APP_ENV: str = "development"
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    INTERNAL_API_KEY: str = "change-me-in-production"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # Phase 3: hardened CV ingestion.
    CV_MAX_FILE_SIZE_BYTES: int = Field(default=10 * 1024 * 1024, gt=0)
    CV_MIN_EXTRACTED_TEXT_CHARS: int = Field(default=80, ge=0)
    CV_MAX_DOCX_UNCOMPRESSED_BYTES: int = Field(default=50 * 1024 * 1024, gt=0)
    CV_MAX_DOCX_MEMBERS: int = Field(default=1_000, gt=0)
    EXTRACTION_MAX_MARKDOWN_CHARS: int = Field(default=100_000, ge=0)

    EXTRACTION_LLM_BASE_URL: str | None = None
    EXTRACTION_LLM_API_KEY: str | None = None
    EXTRACTION_LLM_MODEL: str | None = None
    EXTRACTION_LLM_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0)
    EXTRACTION_LLM_MAX_RETRIES: int = Field(default=2, ge=0, le=10)
    EXTRACTION_LLM_RETRY_BACKOFF_SECONDS: float = Field(default=0.5, ge=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
