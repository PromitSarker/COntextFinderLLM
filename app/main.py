import re
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings, logger
from app.services.pdf_processor import PDFProcessor
from app.services.image_processor import ImageProcessor
from app.services.vector_service import VectorService
from app.services.file_manager import save_file, delete_pdf, is_allowed_file, is_image, is_pdf
from app.services.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    DeleteResponse,
    DocumentMetadata,
    QueryResultItem,
    DocumentCategory,
)
from app.services.gemini_service import GeminiService
from typing import List, Optional
from app.services.document_manager import DocumentManager
import uvicorn


app = FastAPI(title="PDF Semantic Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_manager = DocumentManager()

# Mount static files for PDF access
app.mount("/static", StaticFiles(directory="static"), name="static")

pdf_processor = PDFProcessor()
image_processor = ImageProcessor()
vector_service = VectorService()


@app.get("/categories")
async def list_categories():
    """List all available document categories"""
    try:
        categories = [
            {"value": cat.value, "label": cat.value.replace("_", " ").title()} 
            for cat in DocumentCategory
        ]
        
        return {
            "categories": categories,
            "total_categories": len(categories)
        }
    except Exception as e:
        logger.error(f"Failed to list categories: {str(e)}")
        raise HTTPException(500, f"Failed to list categories: {str(e)}")


@app.post("/upload", response_model=List[UploadResponse])
async def upload_files(
    files: List[UploadFile] = File(...),
    categories: List[DocumentCategory] = Form([DocumentCategory.DEFAULT])
):
    """Upload multiple PDFs or Images with category assignments"""
    if not files:
        raise HTTPException(400, "No files provided")
    
    # Convert enum categories to values
    category_values = [cat.value if isinstance(cat, DocumentCategory) else cat for cat in categories]
    category_values = list(set(category_values))  # Remove duplicates
    
    results = []
    
    for file in files:
        try:
            if not is_allowed_file(file.filename):
                logger.warning(f"Unsupported file type: {file.filename}")
                results.append(UploadResponse(
                    document_id=None,
                    filename=file.filename,
                    chunks_created=0,
                    categories=["error: unsupported file type"]
                ))
                continue
            
            file_content = await file.read()
            file_path = save_file(file_content, file.filename)

            # Process based on file type
            if is_pdf(file.filename):
                pages = pdf_processor.extract_text_with_pages(file_content)
                base_metadata = {
                    "source": file_path,
                    "filename": file.filename,
                    "categories": category_values
                }
                documents = pdf_processor.split_pages(pages, base_metadata)
                
            elif is_image(file.filename):
                documents = image_processor.process_image_with_metadata(file_content, file.filename)
                for doc in documents:
                    doc["metadata"]["categories"] = category_values
            else:
                results.append(UploadResponse(
                    document_id=None,
                    filename=file.filename,
                    chunks_created=0,
                    categories=["error: unsupported file type"]
                ))
                continue
            
            valid_documents = [doc for doc in documents if len(doc["content"]) > 20]
            
            if not valid_documents:
                logger.warning(f"No valid content extracted from {file.filename}")
                results.append(UploadResponse(
                    document_id=None,
                    filename=file.filename,
                    chunks_created=0,
                    categories=category_values
                ))
                continue
            
            doc_ids = await vector_service.add_documents(valid_documents, categories=category_values)
            first_doc_id = doc_ids[0] if doc_ids else None
            
            file_type = "PDF" if is_pdf(file.filename) else "Image"
            logger.info(f"Uploaded {file_type} {file.filename} to categories {category_values} with {len(doc_ids)} chunks")
            
            results.append(UploadResponse(
                document_id=first_doc_id.split("_")[0] if first_doc_id else None,
                filename=file.filename,
                chunks_created=len(doc_ids),
                categories=category_values
            ))
            
        except Exception as e:
            logger.error(f"Upload failed for {file.filename}: {str(e)}", exc_info=True)
            results.append(UploadResponse(
                document_id=None,
                filename=file.filename,
                chunks_created=0,
                categories=[f"error: {str(e)}"]
            ))
    
    return results


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents with optional multi-category filtering"""
    try:
        gemini = GeminiService()
        
        query_categories = None
        if request.categories:
            query_categories = [cat.value if isinstance(cat, DocumentCategory) else cat for cat in request.categories]
        
        normalized_question = request.question
        
        results = await vector_service.query(
            normalized_question, 
            request.top_k,
            categories=query_categories
        )
        
        if not results["ids"] or len(results["ids"][0]) == 0:
            cat_msg = f" in categories {query_categories}" if query_categories else ""
            return QueryResponse(results=[], answer=f"No search results found{cat_msg}")
        
        filtered_results = []
        for idx in range(len(results["ids"][0])):
            metadatas_list = results.get("metadatas", [[]])[0]
            documents_list = results.get("documents", [[]])[0]
            distances_list = results.get("distances", [[]])[0]
            
            if idx >= len(metadatas_list) or idx >= len(documents_list):
                continue
                
            metadata = metadatas_list[idx] or {}
            content = documents_list[idx] or ""
            
            if not content or not content.strip():
                continue
            
            if not all(k in metadata for k in ["source", "filename"]):
                continue
            
            distance = distances_list[idx] if idx < len(distances_list) else 1.0
            similarity_score = 1 / (1 + distance)
            
            if similarity_score < 0.65:
                continue
            
            cleaned_content = await gemini.clean_extracted_text(content)

            if not cleaned_content or not cleaned_content.strip():
                continue

            # Strip any remaining newlines/extra whitespace the model may have left
            cleaned_content = " ".join(cleaned_content.split())
            
            # Handle both old single category and new multiple categories
            doc_categories = metadata.get("categories", [metadata.get("category", "default")])
            if isinstance(doc_categories, str):
                doc_categories = [doc_categories]
            
            filtered_results.append({
                "content": cleaned_content,
                "page_number": metadata.get("page_number", 0),
                "pdf_link": f"{metadata['source']}#page={metadata.get('page_number', 1)}",
                "filename": metadata["filename"],
                "categories": doc_categories
            })
        
        if not filtered_results:
            return QueryResponse(results=[], answer="No relevant results found")
        
        context = "\n\n".join([r['content'] for r in filtered_results[:3]])
        answer = await gemini.answer_question(request.question, context)
        
        if answer == "Not found":
            return QueryResponse(results=[], answer="Unable to generate answer from the available context")
        
        formatted_results = [
            QueryResultItem(
                content=r["content"],
                page_number=r["page_number"],
                pdf_link=r["pdf_link"],
                filename=r["filename"],
                categories=r["categories"]
            )
            for r in filtered_results
        ]
        
        return QueryResponse(results=formatted_results, answer=answer)
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Search failed: {str(e)}")


@app.delete("/document/{filename}", response_model=DeleteResponse)
async def delete_document(filename: str):
    """Delete a document and its vector embeddings"""
    try:
        result = document_manager.delete_document(filename)
        
        return DeleteResponse(
            success=True,
            message=(
                f"Successfully deleted {result['chunks_deleted']} vector chunks. "
                f"Physical file deleted: {result['file_deleted']}"
            )
        )
    except ValueError as ve:
        logger.warning(f"Document not found: {filename} - {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Document deletion failed for {filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@app.delete("/documents/category/{category}", response_model=DeleteResponse)
async def delete_documents_by_category(category: DocumentCategory):
    """Delete all documents in a specific category"""
    try:
        category_value = category.value if isinstance(category, DocumentCategory) else category
        
        result = vector_service.delete_by_category(category_value)
        
        files_deleted = 0
        for filename in result.get("filenames", []):
            try:
                file_path = Path(settings.UPLOAD_DIR) / filename
                if file_path.exists():
                    file_path.unlink()
                    files_deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete file {filename}: {str(e)}")
        
        return DeleteResponse(
            success=True,
            message=(
                f"Successfully deleted {result['chunks_deleted']} vector chunks "
                f"and {files_deleted} physical files from category '{category_value}'"
            )
        )
    except Exception as e:
        logger.error(f"Category deletion failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Category deletion failed: {str(e)}")


@app.delete("/documents/all", response_model=DeleteResponse)
async def delete_all_documents():
    """Delete all documents and their vector embeddings"""
    try:
        # Delete all vector embeddings from the collection
        vector_result = vector_service.delete_all()

        # Delete physical files
        documents_dir = Path(settings.UPLOAD_DIR)
        files_deleted = 0
        if documents_dir.exists():
            for file_path in documents_dir.glob("*"):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        files_deleted += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete file {file_path}: {str(e)}")

        logger.info(f"Deleted all {vector_result['total_deleted']} chunks and {files_deleted} files")

        return DeleteResponse(
            success=True,
            message=f"Successfully deleted all {vector_result['total_deleted']} vector chunks and {files_deleted} physical files"
        )
    except Exception as e:
        logger.error(f"Delete all failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete all failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)