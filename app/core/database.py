
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings

class VectorDB:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = chromadb.PersistentClient(
                path=settings.VECTOR_DB_PATH,
                settings=ChromaSettings(allow_reset=True)
            )
        return cls._instance

# Initialize collection
def get_collection():
    client = VectorDB.get_instance()
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )