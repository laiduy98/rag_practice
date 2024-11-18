import os
from pydantic_settings import BaseSettings

# Define constant here
DATA_PATH = "./data/faqs.json"
SIMILARITY_THRESHOLD = 0.8

class Settings(BaseSettings):
    openai_api_key: str

    class Config:
        env_file = ".env"


# Check if env file is there
if not os.path.exists(".env"):
    raise FileNotFoundError("Environment file is missing.")

settings = Settings()