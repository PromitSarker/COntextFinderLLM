import re

def clean_pdf_text(text: str) -> str:
    """
    Clean PDF text extraction artifacts:
    - Fix line breaks within sentences
    - Preserve paragraph breaks
    - Remove orphaned hyphens
    - Normalize whitespace
    """
    # Step 1: Fix mid-sentence line breaks (preserve paragraph breaks)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single \n -> space
    
    # Step 2: Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Step 3: Fix hyphenated words split across lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)  # "main-\ntenance" -> "maintenance"
    
    # Step 4: Remove orphaned spaces before punctuation
    text = re.sub(r' ([.,!?;:])', r'\1', text)
    
    # Step 5: Normalize paragraph spacing (2+ newlines = paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Step 6: Trim leading/trailing whitespace per paragraph
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return '\n\n'.join(paragraphs)