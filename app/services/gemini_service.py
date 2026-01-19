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
        6. REMOVE PAGE NUMBERS AND HEADER/FOOTER TEXT (e.g., "page 65 66", "| Lesson 4 |", etc.)
        7. REMOVE NAVIGATION BREADCRUMBS AND METADATA (e.g., "| Mix Time Maps | Lesson 4 | Exercise 4 |")
        8. OUTPUT ONLY THE CLEANED TEXT - NO ADDITIONAL COMMENTARY
        
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
    
    async def answer_question(self, question: str, retrieved_context: str) -> str:
        """
        Answer questions ONLY based on retrieved context
        """
        if not retrieved_context.strip():
            return "Not found"
        
        prompt = f"""
        You are a helpful assistant that answers questions STRICTLY based on the provided context.
        
        CRITICAL RULES:
        1. ONLY use information from the context below to answer
        2. If the context doesn't contain relevant information, respond EXACTLY with: "Not found"
        3. DO NOT use your general knowledge or training data
        4. DO NOT make assumptions or inferences beyond the context
        5. If you're uncertain whether the context addresses the question, say "Not found"
        6. HANDLE SPACING VARIATIONS: Treat compound words with and without spaces as equivalent
           Examples: "mixmanufacture" = "mix manufacture", "setpoint" = "set point", "motorcontrol" = "motor control"
        7. BE FLEXIBLE with word boundaries in technical terms while matching against context
        
        Context:
        {retrieved_context}
        
        Question: {question}
        
        Answer (or "Not found" if context is irrelevant):
        """
        
        response = await self.generation_model.generate_content_async(
            prompt,
            generation_config={
                "temperature": 0.1,  # Low temperature for consistency
                "max_output_tokens": 500,
            }
        )
        
        answer = response.text.strip()
        
        # Additional safety check
        if "not found" in answer.lower() or "no information" in answer.lower():
            return "Not found"
        
        return answer

