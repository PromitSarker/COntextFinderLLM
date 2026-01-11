import os
import shutil
from pathlib import Path
from app.core.config import settings

def save_pdf(file_data: bytes, filename: str) -> str:
    """Save PDF to static directory and return relative path"""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    safe_filename = filename.replace(" ", "_").lower()
    file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
    
    with open(file_path, "wb") as f:
        f.write(file_data)
    
    # Use settings.UPLOAD_DIR so return stays accurate if config changes
    return f"/{settings.UPLOAD_DIR}/{safe_filename}"

def delete_pdf(filename: str) -> bool:
    """Delete PDF file from storage"""
    file_path = Path(settings.UPLOAD_DIR) / filename.replace(" ", "_").lower()
    if file_path.exists():
        os.remove(file_path)
        return True
    return False