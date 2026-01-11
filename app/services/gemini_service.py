import google.generativeai as genai
from app.core.config import settings
from typing import List, Dict

genai.configure(api_key=settings.GEMINI_API_KEY)

class GeminiService:
    def __init__(self):
        self.embedding_model = "gemini-embedding-001"
        self.generation_model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        result = genai.embed_content(
            model=self.embedding_model,
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']
    
    async def clean_extracted_text(self, text: str) -> str:
        """
        Fix PDF extraction artifacts using Gemini:
        - Remove random spaces within words
        - Repair split words at line breaks
        - Normalize spacing and punctuation
        - Preserve technical terms and safety warnings
        """
        if not text.strip():
            return text
            
        prompt = f"""
        You are a technical document processor. Clean this text extracted from a service manual:
        
        RULES:
        1. REMOVE ALL RANDOM SPACES WITHIN WORDS (e.g., "d r e a m s" -> "dreams")
        2. FIX HYPHENATED WORDS SPLIT AT LINE BREAKS (e.g., "WARN-\nING" -> "WARNING")
        3. PRESERVE TECHNICAL TERMS, PART NUMBERS, AND SAFETY WARNINGS EXACTLY
        4. KEEP ALL NUMBERS AND SYMBOLS INTACT (e.g., "240V", "M-123A")
        5. MAINTAIN ORIGINAL MEANING - DO NOT SUMMARIZE OR REWRITE
        6. OUTPUT ONLY THE CLEANED TEXT - NO ADDITIONAL COMMENTARY
        
        Original text:
        {text}
        """
        

        response = await self.generation_model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.0,  # Deterministic output
                    "max_output_tokens": 2000,
                    "top_p": 0.95
                },
                safety_settings={
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE"
                }
            )
        return response.text.strip()
        
        