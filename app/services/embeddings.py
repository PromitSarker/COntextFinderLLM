import hashlib

def generate_document_id(source: str, page_number: int, chunk_index: int) -> str:
    """Generate deterministic ID using source, page, and chunk"""
    unique_str = f"{source}_{page_number}_{chunk_index}"
    return hashlib.sha256(unique_str.encode()).hexdigest()[:20]