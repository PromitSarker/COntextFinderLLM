from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class DocumentCategory(str, Enum):
    """Predefined document categories"""
    PLANT_MFG_MAPS = "plant_mfg_maps"
    SUPPLY_NETWORK_MAPS = "supply_network_maps"
    TRANSACTIONAL_MAPS = "transactional_maps"
    PLANT_PROCESSING_MAPS = "plant_processing_maps"
    PLANT_PHARMA_MAP = "plant_pharma_map"
    PLANT_TIRE_MAP = "plant_tire_map"
    VSM_BASICS = "vsm_basics"
    FAST_DRAW = "fast_draw"
    MIX_MAP_MECHANICS = "mix_map_mechanics"
    MAP_MECHANICS = "map_mechanics"
    SPAGHETTI_DIAGRAMS = "spaghetti_diagrams"
    IMPROVEMENTS = "improvements"
    ERP_DATA_IMPORT = "erp_data_import"
    PROJECT_PLANNING = "project_planning"
    INSTALL_LICENSE = "install_license"
    DEFAULT = "default"


class UploadResponse(BaseModel):
    document_id: Optional[str]
    filename: str
    chunks_created: int
    categories: List[str]


class DeleteResponse(BaseModel):
    success: bool
    message: str


class DocumentMetadata(BaseModel):
    source: str
    filename: str
    page_number: Optional[int]
    chunk_index: Optional[int]
    file_path: Optional[str]
    categories: List[str]


class QueryResultItem(BaseModel):
    content: str
    page_number: int
    pdf_link: str
    filename: str
    categories: List[str]


class QueryResponse(BaseModel):
    results: List[QueryResultItem]
    answer: str = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    categories: Optional[List[DocumentCategory]] = None