import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(dotenv_path=BASE_DIR / 'env' / '.env')

AWS_REGION = os.getenv('AWS_REGION')

MODEL = 'eu.anthropic.claude-sonnet-4-20250514-v1:0'
MODEL_TEMPERATURE = 0.3

LLM_READ_TIMEOUT = 300
LLM_CONNECT_TIMEOUT = 60
LLM_MAX_ATTEMPTS = 10
