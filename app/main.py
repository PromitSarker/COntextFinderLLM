import re
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings, logger
from app.services.pdf_processor import PDFProcessor
from app.services.image_processor import ImageProcessor
from app.services.document_processor import DocumentProcessor
from app.services.vector_service import VectorService
from app.services.file_manager import (
    save_file, delete_pdf, is_allowed_file, is_image, is_pdf,
    is_text, is_docx, is_xlsx, is_pptx
)
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
from html2image import Html2Image


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
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent.parent / "static")), name="static")

pdf_processor = PDFProcessor()
image_processor = ImageProcessor()
document_processor = DocumentProcessor()
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


@app.get("/documents")
async def list_documents():
    """List all unique documents and their chunk counts"""
    try:
        return document_manager.list_documents()
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(500, f"Failed to list documents: {str(e)}")


@app.post(
    "/upload",
    response_model=List[UploadResponse],
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "Upload PDF, image, DOCX, PPTX, XLSX, TXT, CSV, and more",
                            },
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "One or more document categories",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def upload_files(
    files: List[UploadFile] = File(..., description="Upload PDF, image, DOCX, PPTX, XLSX, TXT, CSV, and more"),
    categories: List[str] = Form([DocumentCategory.DEFAULT.value], description="One or more document categories")
):
    """Upload multiple files (PDF, images, DOCX, PPTX, XLSX, TXT, CSV, MD, JSON, etc.) with multiple category assignments"""
    if not files:
        raise HTTPException(400, "No files provided")
    
    # Normalize categories: handle JSON-encoded strings, comma-separated values, etc.
    raw_categories = []
    for cat in categories:
        if cat.startswith("["):
            try:
                parsed = json.loads(cat)
                if isinstance(parsed, list):
                    raw_categories.extend(str(c) for c in parsed)
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
        if "," in cat:
            raw_categories.extend(c.strip() for c in cat.split(",") if c.strip())
        else:
            raw_categories.append(cat.strip())
    
    valid_category_names = {c.value for c in DocumentCategory}
    invalid = [c for c in raw_categories if c not in valid_category_names]
    if invalid:
        raise HTTPException(
            400,
            f"Invalid categories: {invalid}. Valid options: {sorted(valid_category_names)}"
        )
    
    category_values = list(set(raw_categories)) if raw_categories else [DocumentCategory.DEFAULT.value]
    
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
            documents = []

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

            elif is_docx(file.filename):
                documents = document_processor.process_docx(file_content, file.filename)
                for doc in documents:
                    doc["metadata"]["source"] = file_path
                    doc["metadata"]["categories"] = category_values

            elif is_pptx(file.filename):
                documents = document_processor.process_pptx(file_content, file.filename)
                for doc in documents:
                    doc["metadata"]["source"] = file_path
                    doc["metadata"]["categories"] = category_values

            elif is_xlsx(file.filename):
                documents = document_processor.process_xlsx(file_content, file.filename)
                for doc in documents:
                    doc["metadata"]["source"] = file_path
                    doc["metadata"]["categories"] = category_values

            elif is_text(file.filename):
                documents = document_processor.process_text_file(file_content, file.filename)
                for doc in documents:
                    doc["metadata"]["source"] = file_path
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
            
            ext = Path(file.filename).suffix.lower()
            logger.info(f"Uploaded {ext} file {file.filename} to categories {category_values} with {len(doc_ids)} chunks")
            
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


@app.post("/upload/url", response_model=UploadResponse)
async def upload_url(
    url: str = Query(..., description="Website URL to ingest (e.g., https://example.com)"),
    categories: List[str] = Query([DocumentCategory.DEFAULT.value], description="One or more document categories")
):
    """
    Takes a snapshot of a URL, uses AI to extract text/images from the screenshot,
    and indexes it into the search database.
    """
    try:
        # Normalize categories: handle JSON-encoded strings, comma-separated values, etc.
        raw_categories = []
        for cat in categories:
            if cat.startswith("["):
                try:
                    parsed = json.loads(cat)
                    if isinstance(parsed, list):
                        raw_categories.extend(str(c) for c in parsed)
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
            if "," in cat:
                raw_categories.extend(c.strip() for c in cat.split(",") if c.strip())
            else:
                raw_categories.append(cat.strip())

        valid_category_names = {c.value for c in DocumentCategory}
        invalid = [c for c in raw_categories if c not in valid_category_names]
        if invalid:
            raise HTTPException(
                400,
                f"Invalid categories: {invalid}. Valid options: {sorted(valid_category_names)}"
            )

        category_values = list(set(raw_categories)) if raw_categories else [DocumentCategory.DEFAULT.value]
        # 1. Take Screenshot
        # Ensure temp dir exists
        Path("./static/temp").mkdir(parents=True, exist_ok=True)
        hti = Html2Image(
            output_path="./static/temp",
            custom_flags=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
        )
        
        screenshot_filename = f"web_{url.replace('://', '_').replace('/', '_')[:50]}.png"
        
        # This captures the screenshot and saves it to ./static/temp/screenshot_filename
        hti.screenshot(url=url, save_as=screenshot_filename, size=(1280, 1600))
        
        file_path = Path(f"./static/temp/{screenshot_filename}")
        
        if not file_path.exists():
            raise HTTPException(400, "Failed to capture website screenshot")
            
        with open(file_path, 'rb') as f:
            image_bytes = f.read()

        # 2. Analyze with Gemini
        logger.info(f"Analyzing screenshot for {url}...")
        extracted_content = await GeminiService().analyze_webpage_screenshot(image_bytes)
        
        if not extracted_content or extracted_content.startswith("Error analyzing screenshot:"):
            raise HTTPException(500, f"AI failed to read website: {extracted_content}")

        # 3. Clean up the temp image (optional, or keep it for reference)
        # file_path.unlink() 

        # 4. Chunk and Store (Treat it like a text file now)
        from app.services.document_processor import DocumentProcessor
        doc_processor = DocumentProcessor()
        
        # Re-use the text splitter
        chunks = doc_processor._split_text(extracted_content, chunk_size=1000, overlap=200)
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "content": chunk,
                "metadata": {
                    "filename": f"URL_{url[:30]}",
                    "source": f"static/temp/{screenshot_filename}",
                    "original_url": url,
                    "page_number": 1,
                    "chunk_index": i,
                    "file_type": "web_url",
                    "categories": category_values
                }
            })
            
        if not documents:
            raise HTTPException(400, "No content extracted from URL")

        # 5. Add to Vector DB
        doc_ids = await vector_service.add_documents(documents, categories=category_values)
        
        return UploadResponse(
            document_id=doc_ids[0].split("_")[0] if doc_ids else None,
            filename=url,
            chunks_created=len(documents),
            categories=category_values
        )

    except Exception as e:
        logger.error(f"URL ingestion failed: {str(e)}")
        raise HTTPException(500, f"Failed to ingest URL: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, http_request: Request):
    """Query documents with optional multi-category filtering"""
    try:
        base_url = str(http_request.base_url).rstrip("/")
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
        seen_filenames = set()

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

            filename = metadata["filename"]

            # Only keep the first (most relevant) chunk per source file
            if filename in seen_filenames:
                continue
            
            distance = distances_list[idx] if idx < len(distances_list) else 1.0
            similarity_score = 1 / (1 + distance)
            
            if similarity_score < 0.65:
                continue
            
            cleaned_content = await gemini.clean_extracted_text(content)

            if not cleaned_content or not cleaned_content.strip():
                continue

            cleaned_content = " ".join(cleaned_content.split())

            seen_filenames.add(filename)
            
            doc_categories = metadata.get("categories", metadata.get("category", "default"))
            if isinstance(doc_categories, str):
                doc_categories = [c.strip() for c in doc_categories.split(",") if c.strip()]
            if not doc_categories:
                doc_categories = ["default"]

            # Prefer the query context over stored doc categories so URLs
            # reflect the context the user is actually browsing.
            if query_categories:
                primary_category = query_categories[0]
            else:
                primary_category = doc_categories[0] if doc_categories else "default"

            # Web URL content: link directly to the original webpage with context
            if metadata.get("file_type") == "web_url":
                original_url = metadata.get("original_url") or metadata.get("source", "")
                pdf_link = f"{original_url}?context={primary_category}"
            else:
                raw_source = metadata.get("source", "")
                if raw_source.startswith("/app/"):
                    url_source = raw_source[len("/app/"):]
                elif raw_source.startswith("/"):
                    url_source = raw_source.lstrip("/")
                else:
                    url_source = raw_source
                url_source = url_source.lstrip("/")

                # Fall back to filename if source is missing or malformed
                if not url_source or "/" not in url_source:
                    url_source = f"static/documents/{Path(metadata['filename']).name}"

                pdf_link = f"{base_url}/{url_source}?context={primary_category}#page={metadata.get('page_number', 1)}"

            filtered_results.append({
                "content": cleaned_content,
                "page_number": metadata.get("page_number", 0),
                "pdf_link": pdf_link,
                "filename": metadata["filename"],
                "categories": doc_categories
            })
        
        if not filtered_results:
            return QueryResponse(results=[], answer="No relevant results found")
        
        context = "\n\n".join([r['content'] for r in filtered_results])
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
        vector_result = vector_service.delete_all()

        files_deleted = 0
        dirs_to_clean = [
            Path(settings.UPLOAD_DIR),
            Path(settings.UPLOAD_DIR).parent / "temp",
        ]
        for target_dir in dirs_to_clean:
            if target_dir.exists():
                for file_path in target_dir.glob("*"):
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