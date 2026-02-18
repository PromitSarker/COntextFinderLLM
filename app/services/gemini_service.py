import google.generativeai as genai
from app.core.config import settings
from typing import List, Dict

genai.configure(api_key=settings.GEMINI_API_KEY)

class GeminiService:
    def __init__(self):
        self.embedding_model = "gemini-embedding-001"
        self.generation_model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    async def get_embeddings(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        result = genai.embed_content(
            model=self.embedding_model,
            content=texts,
            task_type=task_type
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
        
        # Pre-filter: Skip completely garbled text
        if len(text) < 20 or len([c for c in text if c.isalnum()]) / len(text) < 0.3:
            return ""
        
        # Check for repetitive junk patterns
        if any(pattern * 3 in text for pattern in ["Flow %", "Set ", "Day>", "Qty ", "S1S2", "p1 -", "Customer "]):
            return ""
            
        prompt = f"""
        You are a technical document processor. Clean this text extracted from a document.

        JUNK DETECTION — Return exactly "JUNK" if ANY of these apply:
        1. Excessive repetition (same words/patterns repeated 5+ times)
        2. Mostly table fragments, data dumps, or garbled charts
        3. Primarily random variable names without explanation (e.g., "A0010 A0020 A0030")
        4. Just numbers/codes without readable sentences
        5. Alphanumeric ratio suggests corrupted data

        CLEANING RULES (only if text passes junk detection):
        6. Remove ALL newline artifacts — replace \\n, \\r, \\\\n with a single space
        7. Remove ASCII tables (lines with "|", "---", "===", "X X X" patterns)
        8. Remove page numbers (e.g., "Page 1 of 5", "- 3 -", standalone numbers on a line)
        9. Remove headers and footers (e.g., website URLs, copyright lines, document titles repeated at top/bottom)
        10. Remove navigation breadcrumbs (e.g., "Home > Support > Install")
        11. Remove random spaces within words (e.g., "d r e a m s" -> "dreams")
        12. Fix hyphenated words split across lines (e.g., "WARN-\nING" -> "WARNING")
        13. Preserve technical terms, part numbers, and safety warnings exactly
        14. Join all remaining text into clean flowing paragraphs with single spaces between sentences
        15. Do NOT add any commentary, headings, or bullet points that weren't in the original

        OUTPUT FORMAT:
        - One or more clean paragraphs of plain text
        - No bullet symbols, no markdown, no newlines in output
        - Or exactly "JUNK" if the text fails junk detection

        Original text:
        {text}

        OUTPUT:
        """
        
        response = await self.generation_model.generate_content_async(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 2000,
                "top_p": 0.95
            },
            safety_settings={
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE"
            }
        )
        
        cleaned = response.text.strip()
        
        # Post-filter: Reject if marked as junk, too short, or still repetitive
        if (cleaned.upper() == "JUNK" or 
            len(cleaned) < 30 or
            any(word * 4 in cleaned for word in cleaned.split()[:20])):
            return ""
        
        return cleaned

    async def answer_question(self, question: str, retrieved_context: str) -> str:
        """
        Answer questions based on retrieved context with search engine flexibility
        """
        if not retrieved_context.strip():
            return "Not found"
        
        prompt = f"""
        You are a helpful search assistant that answers questions based on the provided document context.
        
        CRITICAL RULES:
        1. If the context contains ONLY garbage/corrupted text (e.g., "S2S1S2 S1S2", random symbols), respond with "Not found"
        2. Answer using ONLY readable, meaningful information from the context
        3. Ignore any garbled, corrupted, or nonsensical text fragments
        4. If the context is SOMEWHAT RELATED but doesn't fully answer the question, provide what information IS available
        5. ONLY respond with "Not found" if the context is COMPLETELY UNRELATED or entirely corrupted
        6. Be flexible with terminology - match semantically similar terms
        7. If context mentions the topic but lacks specific details, say "The document mentions [topic] but doesn't provide [specific detail]"
        8. Keep answers concise, professional, and informative
        9. Never include garbled text or random characters in your answer
        
        Context from documents:
        {retrieved_context}
        
        User Question: {question}
        
        Answer (provide clear information or "Not found" if unrelated/corrupted):
        """
        
        response = await self.generation_model.generate_content_async(
            prompt,
            generation_config={
                "temperature": 0.2,  # Lower for more deterministic filtering
                "max_output_tokens": 500,
            }
        )
        
        answer = response.text.strip()
        
        # Filter junk answers
        if (answer.lower() == "not found" or 
            len(answer) < 10 or 
            len([c for c in answer if c.isalnum()]) / len(answer) < 0.6):
            return "Not found"
        
        return answer

