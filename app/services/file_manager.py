import os
from pathlib import Path
from app.core.config import settings

ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}

def save_file(file_content: bytes, filename: str) -> str:
    """Save uploaded file (PDF or image) to static directory"""
    file_path = Path(settings.UPLOAD_DIR) / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # Return relative path for serving
    return f"static/documents/{filename}"

def save_pdf(file_content: bytes, filename: str) -> str:
    """Backward compatibility - saves PDF"""
    return save_file(file_content, filename)

def delete_pdf(filename: str) -> bool:
    """Delete a file from static directory"""
    file_path = Path(settings.UPLOAD_DIR) / filename
    if file_path.exists():
        file_path.unlink()
        return True
    return False

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def is_image(filename: str) -> bool:
    """Check if file is an image"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    return Path(filename).suffix.lower() in image_extensions

def is_pdf(filename: str) -> bool:
    """Check if file is a PDF"""
    return Path(filename).suffix.lower() == '.pdf'