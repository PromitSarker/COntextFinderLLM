import os
from dotenv import load_dotenv
import logging
from pathlib import Path
import stat

load_dotenv()

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./chroma_db")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./static/documents")
    CHROMA_COLLECTION: str = "document_embeddings"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    def __init__(self):
        # Ensure directories exist with proper permissions
        for directory in [self.VECTOR_DB_PATH, self.UPLOAD_DIR]:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            
            # Set full permissions (read, write, execute)
            try:
                os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 777
                
                # Also ensure parent directory is writable
                if path.parent.exists():
                    os.chmod(path.parent, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    
            except Exception as e:
                print(f"Warning: Could not set permissions on {directory}: {e}")
                
            # Verify directory is writable
            if not os.access(path, os.W_OK):
                print(f"ERROR: Directory {path} is not writable!")
                raise PermissionError(f"Cannot write to {path}")
    
    # Validation
    def validate(self):
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in environment variables")

settings = Settings()
settings.validate()

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("pdf-search")