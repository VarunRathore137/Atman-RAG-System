from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from config import cfg
from config.logging_config import logger
from src.chunking.models import EnrichedChunk


class ChromaVectorStore:
    """
    Persistent ChromaDB vector store with cosine distance space.

    Why cosine distance (not L2/dot-product):
    - Chunks vary in length (120–800 chars). Cosine is length-invariant,
      so a short FAQ answer scores equally to a longer policy paragraph
      if the semantic content matches the query.
    - Cosine distance d in [0, 2]; similarity = 1 - d (for normalized vectors).

    ChromaDB stores vectors in an HNSW index backed by a local SQLite file
    at cfg.CHROMA_DB_DIR. No external service, no network, fully offline.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        db_path = str(persist_dir or cfg.CHROMA_DB_DIR)
        self.collection_name = collection_name or cfg.CHROMA_COLLECTION

        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB ready: collection='{self.collection_name}' "
            f"path='{db_path}' count={self.count()}"
        )

    @property
    def collection(self):
        """Return the underlying ChromaDB Collection instance."""
        return self._collection

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        return self._collection.count()

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve records from ChromaDB matching filters.
        """
        kwargs: Dict[str, Any] = {}
        if ids is not None:
            kwargs["ids"] = ids
        if where is not None:
            kwargs["where"] = where
        if limit is not None:
            kwargs["limit"] = limit
        if offset is not None:
            kwargs["offset"] = offset
        if include is not None:
            kwargs["include"] = include
        return self._collection.get(**kwargs)

    def upsert_chunks(
        self,
        chunks: List[EnrichedChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Upsert chunks into ChromaDB.
        Upsert (not add) ensures idempotency — re-running with same chunk_ids
        updates existing records rather than creating duplicates.

        Returns the number of vectors now in the collection.
        """
        if not chunks or not embeddings:
            return self.count()

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.to_chroma_metadata() for c in chunks]

        # Chroma upsert in one batch for efficiency
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        total = self.count()
        logger.info(f"Upserted {len(chunks)} chunks. Collection total: {total}")
        return total

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store by embedding similarity.

        Returns ChromaDB result dict with keys:
          ids, documents, metadatas, distances
        Distances are cosine distances in [0, 2]; lower = more similar.
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        return self._collection.query(**kwargs)

    def reset_collection(self) -> None:
        """
        Delete and recreate the collection for a clean --force reindex.
        Called by CorpusIndexer when force_reindex=True.
        """
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{self.collection_name}' reset. Count: {self.count()}")
