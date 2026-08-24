from typing import Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

# Maps each PDF doc_name to its processing category.
# This drives which splitter SemanticChunker uses per document.
DOC_TYPE_MAP: Dict[str, str] = {
    "Pricing_and_SLA": "data_tables",
    "Product_Manual": "structured_manual",
    "API_Reference": "technical_api",
    "Employee_Handbook": "narrative_policy",
    "Security_Policy": "policy_defined_terms",
    "Onboarding_Guide": "chronological_guide",
    "FAQ_Support": "qa_pairs",
}


class EnrichedChunk(BaseModel):
    """
    A semantically enriched chunk ready for embedding and ChromaDB storage.
    Carries full provenance metadata so every retrieved chunk can be cited.
    """
    chunk_id: str
    doc_name: str
    doc_filename: str
    doc_code: str = "UNKNOWN"
    doc_version: str = "1.0"
    page_num: int
    section_heading: str = "General"
    chunk_type: str = "text"      # 'text' | 'table' | 'code' | 'qa_pair'
    text: str
    char_count: int
    token_count_est: int
    has_table: bool = False
    has_code: bool = False
    doc_type: str = "general"
    ingestion_ts: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """
        Return a flat dict with ONLY scalar values (str/int/float/bool).
        ChromaDB rejects nested objects — the 'metadata' dict is excluded here.
        """
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "doc_filename": self.doc_filename,
            "doc_code": self.doc_code,
            "doc_version": self.doc_version,
            "page_num": self.page_num,
            "section_heading": self.section_heading,
            "chunk_type": self.chunk_type,
            "char_count": self.char_count,
            "token_count_est": self.token_count_est,
            "has_table": self.has_table,
            "has_code": self.has_code,
            "doc_type": self.doc_type,
            "ingestion_ts": self.ingestion_ts,
        }
