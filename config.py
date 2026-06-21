from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    OPENAI_API_KEY: Optional[str]
    PINECONE_API_KEY: Optional[str]
    TOKEN_SECRET: Optional[str]
    TAVILY_API_KEY: Optional[str]
       

    """Loads the dotenv file. Including this is necessary to get
    pydantic to load a .env file."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # model_config = SettingsConfigDict(extra="ignore")