import os
from pathlib import Path
from app.core.config import settings, logger

ALLOWED_EXTENSIONS = {
    # PDFs
    ".pdf",
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp",
    # Documents
    ".docx", ".doc", ".txt", ".rtf", ".odt",
    # Spreadsheets
    ".xlsx", ".xls", ".csv",
    # Presentations
    ".pptx", ".ppt",
    # Markdown / Web
    ".md", ".html", ".htm",
    # JSON / XML
    ".json", ".xml",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".rtf"}
DOCX_EXTENSIONS = {".docx", ".doc", ".odt"}
XLSX_EXTENSIONS = {".xlsx", ".xls"}
PPTX_EXTENSIONS = {".pptx", ".ppt"}


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_allowed_file(filename: str) -> bool:
    return get_extension(filename) in ALLOWED_EXTENSIONS


def is_image(filename: str) -> bool:
    return get_extension(filename) in IMAGE_EXTENSIONS


def is_pdf(filename: str) -> bool:
    return get_extension(filename) in PDF_EXTENSIONS


def is_text(filename: str) -> bool:
    return get_extension(filename) in TEXT_EXTENSIONS


def is_docx(filename: str) -> bool:
    return get_extension(filename) in DOCX_EXTENSIONS


def is_xlsx(filename: str) -> bool:
    return get_extension(filename) in XLSX_EXTENSIONS


def is_pptx(filename: str) -> bool:
    return get_extension(filename) in PPTX_EXTENSIONS


def save_file(content: bytes, filename: str) -> str:
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename
    file_path.write_bytes(content)
    logger.info(f"Saved file: {file_path}")
    return str(file_path)


def delete_pdf(filename: str) -> bool:
    file_path = Path(settings.UPLOAD_DIR) / filename
    if file_path.exists():
        file_path.unlink()
        logger.info(f"Deleted file: {file_path}")
        return True
    return False