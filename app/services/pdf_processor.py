import PyPDF2
from io import BytesIO
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.text_cleaner import clean_pdf_text


class PDFProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        # Optimized for technical manuals
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
            is_separator_regex=False
        )
    
    def extract_text_with_pages(self, pdf_bytes: bytes) -> List[Dict]:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        pages = []
        
        for i, page in enumerate(pdf_reader.pages):
            raw_text = page.extract_text()
            if not raw_text:
                continue
                
            # CRITICAL: Clean text before processing
            cleaned_text = clean_pdf_text(raw_text)
            
            if cleaned_text.strip():  # Skip empty pages
                pages.append({
                    "page_number": i+1,
                    "text": cleaned_text
                })
        return pages
    
    def split_pages(self, pages: List[Dict], metadata: Optional[Dict] = None) -> List[Dict]:
        """Split pages into chunks and ensure metadata is always a dict."""
        metadata = metadata or {}
        chunks = []
        for page in pages:
            # Split by paragraphs first
            paragraphs = [p.strip() for p in page["text"].split('\n\n') if p.strip()]
            
            for para in paragraphs:
                if len(para) < 50:  # Skip very short paragraphs
                    continue
                    
                # Further split long paragraphs
                para_chunks = self.text_splitter.split_text(para)
                for idx, chunk in enumerate(para_chunks):
                    if len(chunk) < 30:  # Skip tiny fragments
                        continue
                        
                    chunks.append({
                        "content": chunk.strip(),
                        "metadata": {
                            **metadata,
                            "page_number": page["page_number"],
                            "chunk_index": idx,
                            "total_chunks": len(para_chunks)
                        }
                    })
        return chunks