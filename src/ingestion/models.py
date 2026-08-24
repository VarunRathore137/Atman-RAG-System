from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ExtractedTable(BaseModel):
    table_id: str
    page_num: int
    markdown_content: str
    row_count: int
    col_count: int
    raw_headers: List[str] = Field(default_factory=list)

class PageDocument(BaseModel):
    doc_name: str
    doc_filename: str
    doc_code: Optional[str] = None
    doc_version: Optional[str] = None
    page_num: int
    text: str
    tables: List[ExtractedTable] = Field(default_factory=list)
    has_tables: bool = False
    extractor_used: str = "pdfplumber"
    metadata: Dict[str, Any] = Field(default_factory=dict)
