from pathlib import Path
from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Config:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DOCS_DIR: Path = PROJECT_ROOT / os.getenv('DOCS_DIR', 'files')
    CHROMA_DB_DIR: Path = PROJECT_ROOT / os.getenv('CHROMA_DB_DIR', 'chroma_db')
    CHROMA_COLLECTION: str = os.getenv('CHROMA_COLLECTION', 'atman_docs')

    TEXT_CHUNK_SIZE: int = int(os.getenv('TEXT_CHUNK_SIZE', 800))
    TEXT_CHUNK_OVERLAP: int = int(os.getenv('TEXT_CHUNK_OVERLAP', 150))
    CODE_CHUNK_SIZE: int = int(os.getenv('CODE_CHUNK_SIZE', 1000))
    CODE_CHUNK_OVERLAP: int = int(os.getenv('CODE_CHUNK_OVERLAP', 0))
    TABLE_CHUNK_MAX_SIZE: int = int(os.getenv('TABLE_CHUNK_MAX_SIZE', 3000))

    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDING_BATCH_SIZE: int = int(os.getenv('EMBEDDING_BATCH_SIZE', 64))

    RETRIEVAL_K: int = int(os.getenv('RETRIEVAL_K', 20))
    RETRIEVAL_THRESHOLD: float = float(os.getenv('RETRIEVAL_THRESHOLD', 0.25))
    RERANK_TOP_N: int = int(os.getenv('RERANK_TOP_N', 5))
    RERANKER_MODEL: str = os.getenv('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    VECTOR_WEIGHT: float = float(os.getenv('VECTOR_WEIGHT', 0.35))
    RERANK_WEIGHT: float = float(os.getenv('RERANK_WEIGHT', 0.65))

    CONFIDENCE_THRESHOLD: float = float(os.getenv('CONFIDENCE_THRESHOLD', 0.40))
    ABSTENTION_PHRASE: str = (
        'I don\'t have enough information in the provided documents to answer this question.'
    )

    LLM_PROVIDER: Literal['groq', 'ollama', 'gemini'] = os.getenv('LLM_PROVIDER', 'groq').lower()
    
    GROQ_API_KEY: str = os.getenv('GROQ_API_KEY', '')
    GROQ_MODEL: str = os.getenv('GROQ_MODEL', 'groq/compound-mini')

    OLLAMA_BASE_URL: str = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL: str = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')

    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')

    LLM_TEMPERATURE: float = float(os.getenv('LLM_TEMPERATURE', 0.0))
    LLM_MAX_TOKENS: int = int(os.getenv('LLM_MAX_TOKENS', 1024))
    MEMORY_TURNS: int = int(os.getenv('MEMORY_TURNS', 4))

cfg = Config()
