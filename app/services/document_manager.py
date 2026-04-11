import logging
from pathlib import Path
from app.core.database import get_collection
from app.services.file_manager import delete_pdf
from app.core.config import settings

logger = logging.getLogger("document-manager")

class DocumentManager:
    def __init__(self):
        self.collection = get_collection()
    
    def delete_document(self, filename: str) -> dict:
        """Atomic document deletion with verification"""
        # 1. Verify document exists in vector DB — try exact filename match first,
        #    then fall back to stem match in case the caller omitted the extension.
        existing = self.collection.get(
            where={"filename": filename},
            include=["metadatas"]
        )

        actual_filename = filename
        if not existing["ids"]:
            # Try matching by stem (e.g. "systemreq" → "systemreq.png")
            query_stem = Path(filename).stem
            all_docs = self.collection.get(include=["metadatas"])
            matched_filename = None
            for meta in (all_docs.get("metadatas") or []):
                stored = (meta or {}).get("filename", "")
                if stored and Path(stored).stem == query_stem:
                    matched_filename = stored
                    break

            if not matched_filename:
                raise ValueError(f"Document not found in vector database: {filename}")

            existing = self.collection.get(
                where={"filename": matched_filename},
                include=["metadatas"]
            )
            actual_filename = matched_filename
        
        # 2. Delete from vector DB FIRST
        try:
            self.collection.delete(ids=existing["ids"])
        except Exception as e:
            raise RuntimeError(f"Vector database deletion failed: {str(e)}")
        
        # 3. Verify deletion succeeded
        verification = self.collection.get(
            where={"filename": actual_filename},
            include=[]
        )
        if verification["ids"]:
            raise RuntimeError(
                f"Deletion verification failed: {len(verification['ids'])} chunks remain"
            )
        
        # 4. Delete physical file
        file_deleted = delete_pdf(actual_filename)
        if not file_deleted:
            logger.warning(
                f"Physical file deletion failed for {actual_filename}, "
                f"but vector database entries were cleaned"
            )
        
        return {
            "chunks_deleted": len(existing["ids"]),
            "file_deleted": file_deleted,
            "filename": actual_filename
        }

    def list_documents(self) -> list:
        """List all unique documents and their chunk counts"""
        try:
            results = self.collection.get(include=["metadatas"])
            if not results["metadatas"]:
                return []
            
            doc_stats = {}
            for meta in results["metadatas"]:
                if not meta: continue
                fname = meta.get("filename", "unknown")
                if fname not in doc_stats:
                    doc_stats[fname] = {
                        "filename": fname,
                        "chunk_count": 0,
                        "categories": meta.get("categories", ["default"]),
                        "type": meta.get("file_type", "document")
                    }
                doc_stats[fname]["chunk_count"] += 1
            
            return sorted(list(doc_stats.values()), key=lambda x: x["filename"])
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            return []