import os
from dotenv import load_dotenv
import logging

load_dotenv()

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./chroma_db")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./static/documents")  # default added
    CHROMA_COLLECTION: str = "document_embeddings"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Validation
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment variables")

settings = Settings()

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("pdf-search")