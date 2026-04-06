# from pydantic_settings import BaseSettings, SettingsConfigDict
# from typing import Optional

# class Settings(BaseSettings):
#     mongodb_url: str = "mongodb://localhost:27017"
#     mongodb_db_name: str = "test"
#     google_api_key: str = "AIzaSyASd6DpLSeBCLqEwVBZn_fRJ5jwtZIsh4g"  # Use env var in production

#     model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from .env file
    All sensitive and configurable values are now coming from .env
    """

    # MongoDB Configuration
    mongodb_url: str
    mongodb_db_name: str

    # Google Gemini Configuration
    google_api_key: str
    gemini_model: str = "gemini-2.5-flash"   # Default value if not in .env

    # Optional settings
    environment: str = "production"
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

# Global settings instance
settings = Settings()