from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "test"
    google_api_key: str = "AIzaSyASd6DpLSeBCLqEwVBZn_fRJ5jwtZIsh4g"  # Use env var in production

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()