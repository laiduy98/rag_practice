import os
from pydantic_settings import BaseSettings

DATA_PATH = "./data/faqs.json"
SIMILARITY_THRESHOLD = 0.8

class Settings(BaseSettings):
    openai_api_key: str

    class Config:
        env_file = ".env"

settings = Settings()