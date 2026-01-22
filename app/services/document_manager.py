import logging
from pathlib import Path
from app.core.database import get_collection
from app.services.file_manager import delete_pdf
from app.core.config import settings

logger = logging.getLogger("document-manager")

class DocumentManager:
    def __init__(self):
        self.collection = get_collection()
    
    def get_document_source(self, filename: str) -> str:
        """Get normalized source path matching upload logic"""
        # Use SAME normalization as in main.py upload endpoint
        safe_filename = filename.replace(" ", "_").lower()
        return f"/static/documents/{Path(safe_filename).name}"
    
    def delete_document(self, filename: str) -> dict:
        """Atomic document deletion with verification"""
        source_path = self.get_document_source(filename)
        
        # 1. Verify document exists in vector DB
        existing = self.collection.get(
            where={"source": source_path},
            include=["metadatas"]
        )
        
        if not existing["ids"]:
            raise ValueError(f"Document not found in vector database: {filename}")
        
        # 2. Delete from vector DB FIRST
        try:
            self.collection.delete(ids=existing["ids"])
        except Exception as e:
            raise RuntimeError(f"Vector database deletion failed: {str(e)}")
        
        # 3. Verify deletion succeeded
        verification = self.collection.get(
            where={"source": source_path},
            include=[]
        )
        if verification["ids"]:
            raise RuntimeError(
                f"Deletion verification failed: {len(verification['ids'])} chunks remain"
            )
        
        # 4. Delete physical file
        file_deleted = delete_pdf(filename)
        if not file_deleted:
            logger.warning(
                f"Physical file deletion failed for {filename}, "
                f"but vector database entries were cleaned"
            )
        
        return {
            "chunks_deleted": len(existing["ids"]),
            "file_deleted": file_deleted,
            "source_path": source_path,
            "filename": filename
        }