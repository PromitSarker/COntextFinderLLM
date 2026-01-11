from pydantic import BaseModel
from typing import List, Optional

class UploadResponse(BaseModel):
    document_id: Optional[str]
    filename: str
    chunks_created: int

class DeleteResponse(BaseModel):
    success: bool
    message: str

class DocumentMetadata(BaseModel):
    source: str
    filename: str
    page_number: Optional[int]
    chunk_index: Optional[int]
    file_path: Optional[str]

class QueryResultItem(BaseModel):
    content: str
    page_number: int
    pdf_link: str
    filename: str

class QueryResponse(BaseModel):
    results: List[QueryResultItem] = []

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5