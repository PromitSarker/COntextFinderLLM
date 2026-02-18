import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
import os
import subprocess

class VectorDB:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            db_path = settings.VECTOR_DB_PATH
            os.makedirs(db_path, exist_ok=True)
            
            # Fix permissions using subprocess (most reliable in Docker)
            try:
                subprocess.run(['chmod', '-R', '777', db_path], check=False)
            except Exception as e:
                print(f"Warning: Could not set permissions via subprocess: {e}")
            
            # Also fix via Python
            try:
                for root, dirs, files in os.walk(db_path):
                    os.chmod(root, 0o777)
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o777)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o666)
            except Exception as e:
                print(f"Warning: Could not set permissions via os.chmod: {e}")
            
            cls._instance = chromadb.PersistentClient(
                path=db_path,
                settings=ChromaSettings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            print(f"ChromaDB initialized at: {db_path}")
        return cls._instance

# Initialize collection — embedding_function=None because we pass Gemini embeddings explicitly
def get_collection():
    client = VectorDB.get_instance()
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        embedding_function=None,  # Disables ChromaDB's ONNX model entirely
        metadata={"hnsw:space": "cosine"}
    )