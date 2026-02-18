import google.generativeai as genai
from app.core.config import settings, logger
from typing import List, Dict
import tempfile
import os

class ImageProcessor:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def extract_text_from_image(self, image_bytes: bytes, filename: str) -> str:
        """Extract text from image using Gemini Vision without PIL"""
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Upload file to Gemini
                uploaded_file = genai.upload_file(tmp_path)
                
                prompt = """
                Extract all text from this image. 
                - Preserve the layout and structure
                - Include any tables, diagrams labels, or technical information
                - If there are multiple sections, separate them clearly
                - Return only the extracted text without any additional commentary
                """
                
                response = self.model.generate_content([prompt, uploaded_file])
                text = response.text.strip()
                
                logger.info(f"Extracted {len(text)} characters from image using Gemini Vision")
                return text
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Image text extraction failed: {str(e)}")
            raise
    
    def process_image_with_metadata(self, image_bytes: bytes, filename: str) -> List[Dict]:
        """Process image and return documents with metadata"""
        try:
            text = self.extract_text_from_image(image_bytes, filename)
            
            if not text or len(text) < 20:
                logger.warning(f"Insufficient text extracted from {filename}")
                return []
            
            # Split into chunks if text is very long
            max_chunk_size = 2000
            chunks = []
            
            if len(text) <= max_chunk_size:
                chunks = [text]
            else:
                paragraphs = text.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) <= max_chunk_size:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + "\n\n"
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
            
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append({
                    "content": chunk,
                    "metadata": {
                        "source": f"static/documents/{filename}",
                        "filename": filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_type": "image"
                    }
                })
            
            logger.info(f"Processed image {filename} into {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Image processing failed for {filename}: {str(e)}")
            raise