import uuid
from typing import List, Dict
from app.core.database import get_collection
from app.core.config import logger
from app.services.gemini_service import GeminiService
from app.services.pdf_processor import PDFProcessor
from app.services.embeddings import generate_document_id


class VectorService:
    def __init__(self):
        self.collection = get_collection()
        self.gemini = GeminiService()
        self.pdf_processor = PDFProcessor()

    async def add_documents(self, documents: List[Dict]) -> List[str]:
        """Add pre-processed documents to vector DB. Validates inputs & embedding lengths."""
        if not documents:
            logger.debug("add_documents called with empty documents list")
            return []

        ids = [
            generate_document_id(
                doc["metadata"]["source"],
                doc["metadata"].get("page_number"),
                doc["metadata"].get("chunk_index"),
            )
            for doc in documents
        ]

        contents = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        try:
            embeddings = await self.gemini.get_embeddings(contents)
        except Exception as e:
            logger.error("Failed to get embeddings: %s", str(e), exc_info=True)
            raise

        if len(embeddings) != len(contents):
            logger.error(
                "Embedding count mismatch: %d embeddings for %d documents",
                len(embeddings),
                len(contents),
            )
            raise ValueError("Embedding service returned unexpected number of vectors")

        # Add to Chroma (synchronous)
        self.collection.add(ids=ids, embeddings=embeddings, documents=contents, metadatas=metadatas)
        return ids

    async def process_and_add_pdf(self, file_path: str, file_name: str) -> List[str]:
        """Process PDF file and add chunks to vector DB"""
        try:
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            pages = self.pdf_processor.extract_text_with_pages(pdf_bytes)
            base_metadata = {"source": file_name, "file_path": file_path}
            chunks = self.pdf_processor.split_pages(pages, metadata=base_metadata)

            return await self.add_documents(chunks)
        except Exception as e:
            logger.error("Failed to process and add PDF: %s", str(e), exc_info=True)
            raise

    def delete_document(self, source: str) -> dict:
        """Delete all chunks of a document by source filename and return deletion summary."""
        results = self.collection.get(where={"source": source}, include=["metadatas", "ids"])
        ids = results.get("ids", [])

        # Handle nested list shape that Chroma may return
        if ids and isinstance(ids[0], list):
            ids = ids[0]

        if not ids:
            return {"chunks_deleted": 0, "ids": []}

        self.collection.delete(ids=ids)
        return {"chunks_deleted": len(ids), "ids": ids}

    def delete_all(self) -> dict:
        """Delete all documents and embeddings from vector database."""
        try:
            total_count = self.collection.count()
            if total_count > 0:
                # Get all IDs and delete them
                all_data = self.collection.get(include=[])
                all_ids = all_data.get("ids", [])
                if all_ids:
                    self.collection.delete(ids=all_ids)
            return {"total_deleted": total_count}
        except Exception as e:
            logger.error("Delete all from vector database failed: %s", str(e), exc_info=True)
            raise

    async def query(self, query_text: str, top_k: int = 5) -> dict:
        """Returns raw ChromaDB results with safety checks."""
        try:
            query_embedding = (await self.gemini.get_embeddings([query_text]))[0]
            max_k = min(top_k, max(1, self.collection.count()))

            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Vector query failed: %s", str(e), exc_info=True)
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    async def search_similar(self, query: str, threshold: float = 0.7, top_k: int = 3) -> List[Dict]:
        """
        Search with minimum similarity threshold
        Returns empty list if no results meet threshold
        """
        results = await self.query(query, top_k=top_k)
        
        if not results["ids"] or len(results["ids"][0]) == 0:
            return []
        
        # Calculate similarity scores and filter
        filtered_results = []
        for idx in range(len(results["ids"][0])):
            # Similarity score from distance (lower distance = higher similarity)
            distance = results.get("distances", [[]])[0][idx] if results.get("distances") else 1.0
            similarity_score = 1 / (1 + distance)  # Convert distance to similarity
            
            if similarity_score < threshold:
                continue
            
            metadata = results.get("metadatas", [[]])[0][idx] or {}
            content = results.get("documents", [[]])[0][idx] or ""
            
            filtered_results.append({
                "text": content,
                "similarity_score": similarity_score,
                "metadata": metadata
            })
        
        return filtered_results