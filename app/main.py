import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from app.core.config import settings, logger
from app.services.pdf_processor import PDFProcessor
from app.services.vector_service import VectorService
from app.services.file_manager import save_pdf, delete_pdf
from app.services.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    DeleteResponse,
    DocumentMetadata,
    QueryResultItem,  # added
)
from app.services.gemini_service import GeminiService
from typing import List
from app.services.document_manager import DocumentManager


app = FastAPI(title="PDF Semantic Search API")
document_manager = DocumentManager()

# Mount static files for PDF access
app.mount("/static", StaticFiles(directory="static"), name="static")

pdf_processor = PDFProcessor()
vector_service = VectorService()

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")
    
    try:
        file_content = await file.read()
        file_path = save_pdf(file_content, file.filename)

        # EXTRACT WITH CLEANING
        pages = pdf_processor.extract_text_with_pages(file_content)
        documents = pdf_processor.split_pages(pages, {
            "source": file_path,
            "filename": file.filename
        })
        
        # SANITY CHECK: Skip empty documents
        valid_documents = [doc for doc in documents if len(doc["content"]) > 20]
        
        if not valid_documents:
            return UploadResponse(
                document_id=None,
                filename=file.filename,
                chunks_created=0
            )
        
        doc_ids = await vector_service.add_documents(valid_documents)
        first_doc_id = doc_ids[0] if doc_ids else None
        return UploadResponse(
            document_id=first_doc_id.split("_")[0] if first_doc_id else None,
            filename=file.filename,
            chunks_created=len(documents)
        )
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.delete("/document/{filename}", response_model=DeleteResponse)
async def delete_document(filename: str):
    """
    Delete a document and its vector embeddings atomically
    - First deletes from vector database
    - Then deletes physical file
    - Verifies complete removal from vector database
    """
    try:
        # Atomic deletion with verification
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
        raise HTTPException(
            status_code=404,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Document deletion failed for {filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Deletion failed: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        results = await vector_service.query(request.question, request.top_k)
        
        if not results["ids"] or len(results["ids"][0]) == 0:
            return QueryResponse(results=[])
        
        formatted_results = []
        gemini = GeminiService() 

        # Safely iterate through results
        for idx in range(len(results["ids"][0])):
            # Defensive metadata access
            metadatas_list = results.get("metadatas", [[]])[0]
            documents_list = results.get("documents", [[]])[0]
            
            if idx >= len(metadatas_list) or idx >= len(documents_list):
                continue
                
            metadata = metadatas_list[idx] or {}
            content = documents_list[idx] or ""
            
            # Skip if critical metadata is missing
            if not all(k in metadata for k in ["page_number", "source", "filename"]):
                continue
                
            raw_content = documents_list[idx] or ""
            
            # CRITICAL: Clean text using Gemini BEFORE returning
            cleaned_content = await gemini.clean_extracted_text(raw_content)
            
            formatted_results.append(QueryResultItem(
                content=cleaned_content,  # CLEANED VERSION
                page_number=metadata["page_number"],
                pdf_link=f"{metadata['source']}#page={metadata['page_number']}",
                filename=metadata["filename"]
            ))
        
        return QueryResponse(results=formatted_results)
        
    except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            raise HTTPException(500, f"Search failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)