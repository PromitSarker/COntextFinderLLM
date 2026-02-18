from app.core.config import settings, logger
from app.core.database import get_collection
from app.services.gemini_service import GeminiService
from typing import List, Dict, Optional
import uuid


class VectorService:
    def __init__(self):
        self.collection = get_collection()

    async def add_documents(self, documents: List[Dict], categories: Optional[List[str]] = None) -> List[str]:
        """Add documents using Gemini embeddings (bypasses ChromaDB ONNX model)"""
        if not documents:
            logger.debug("add_documents called with empty documents list")
            return []

        ids = []
        contents = []
        metadatas = []

        for i, doc in enumerate(documents):
            ids.append(f"{uuid.uuid4().hex}_{i}")
            contents.append(doc["content"])

            metadata = doc.get("metadata", {}).copy()

            # Resolve category list
            if categories:
                cat_list = categories
            elif "categories" in metadata:
                cat_list = metadata["categories"] if isinstance(metadata["categories"], list) else [metadata["categories"]]
            else:
                cat_list = ["default"]

            # Flatten metadata to ChromaDB-compatible types (str/int/float/bool only)
            clean_metadata = {}
            for k, v in metadata.items():
                if v is None or k == "categories":
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                elif isinstance(v, list):
                    clean_metadata[k] = ",".join(str(x) for x in v)
                else:
                    clean_metadata[k] = str(v)

            # Store categories as individual boolean flags for reliable $eq filtering
            # e.g. cat_install_license=True, cat_vsm_basics=True
            clean_metadata["categories"] = ",".join(cat_list)  # human-readable summary
            for cat in cat_list:
                clean_metadata[f"cat_{cat}"] = True

            metadatas.append(clean_metadata)

        # Generate embeddings with Gemini (no ONNX model needed)
        gemini = GeminiService()
        embeddings = await gemini.get_embeddings(contents)

        # Insert in batches
        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=contents[start:end],
                metadatas=metadatas[start:end]
            )

        logger.info(f"Added {len(ids)} documents with Gemini embeddings (categories: {categories or ['default']})")
        return ids

    async def query(self, question: str, top_k: int = 5, categories: Optional[List[str]] = None) -> Dict:
        """Query using Gemini embeddings (bypasses ChromaDB ONNX model)"""
        try:
            collection_count = self.collection.count()
            logger.info(f"Collection has {collection_count} documents")

            if collection_count == 0:
                logger.warning("Querying empty collection")
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            max_k = min(top_k, collection_count)

            # Embed the query with Gemini using retrieval_query task type
            gemini = GeminiService()
            query_embedding = await gemini.get_embeddings([question], task_type="retrieval_query")

            query_params = {
                "query_embeddings": query_embedding,
                "n_results": max_k,
                "include": ["documents", "metadatas", "distances"]
            }

            # Filter using boolean category flags — $eq is supported in all ChromaDB versions
            if categories:
                if len(categories) == 1:
                    query_params["where"] = {f"cat_{categories[0]}": {"$eq": True}}
                else:
                    query_params["where"] = {
                        "$or": [{f"cat_{cat}": {"$eq": True}} for cat in categories]
                    }
                logger.info(f"Querying with category filter: {categories}")

            results = self.collection.query(**query_params)
            logger.info(f"Query returned {len(results['ids'][0])} results")
            return results

        except Exception as e:
            logger.error(f"Vector query failed: {str(e)}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def delete_by_filename(self, filename: str) -> int:
        """Delete all chunks for a given filename."""
        try:
            results = self.collection.get(
                where={"filename": filename},
                include=[]
            )
            
            if not results["ids"]:
                logger.warning(f"No documents found for filename: {filename}")
                return 0
            
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks for {filename}")
            return len(results["ids"])
        except Exception as e:
            logger.error(f"Delete by filename failed: {str(e)}")
            raise

    def delete_by_category(self, category: str) -> Dict:
        """Delete all documents in a specific category."""
        try:
            results = self.collection.get(
                where={f"cat_{category}": {"$eq": True}},
                include=["metadatas"]
            )
            
            if not results["ids"]:
                return {"chunks_deleted": 0, "filenames": []}
            
            filenames = list(set(
                m.get("filename", "") 
                for m in results["metadatas"] 
                if m and m.get("filename")
            ))
            
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks from category '{category}'")
            
            return {
                "chunks_deleted": len(results["ids"]),
                "filenames": filenames
            }
        except Exception as e:
            logger.error(f"Delete by category failed: {str(e)}")
            raise

    def delete_all(self) -> Dict:
        """Delete all documents from the vector store."""
        try:
            all_docs = self.collection.get(include=[])
            total = len(all_docs["ids"]) if all_docs["ids"] else 0
            
            if total > 0:
                self.collection.delete(ids=all_docs["ids"])
            
            logger.info(f"Deleted all {total} documents from vector store")
            return {"total_deleted": total}
        except Exception as e:
            logger.error(f"Delete all failed: {str(e)}")
            raise

    def get_documents_by_category(self, category: str) -> Dict:
        """Get all documents in a specific category."""
        try:
            results = self.collection.get(
                where={f"cat_{category}": {"$eq": True}},
                include=["documents", "metadatas"]
            )
            return results
        except Exception as e:
            logger.error(f"Get documents by category failed: {str(e)}")
            raise

    def list_categories(self) -> List[str]:
        """List all unique categories in the collection."""
        try:
            results = self.collection.get(include=["metadatas"])
            categories = set()

            for metadata in results["metadatas"]:
                if metadata and "categories" in metadata:
                    # Parse comma-separated categories string
                    for cat in metadata["categories"].split(","):
                        cat = cat.strip()
                        if cat:
                            categories.add(cat)

            return sorted(categories)
        except Exception as e:
            logger.error(f"List categories failed: {str(e)}")
            raise