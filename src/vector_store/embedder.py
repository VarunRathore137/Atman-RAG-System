from typing import List, Optional

from sentence_transformers import SentenceTransformer

from config import cfg
from config.logging_config import logger


class EmbeddingEngine:
    """
    Wraps SentenceTransformer for batch and single-query dense embeddings.

    Model: sentence-transformers/all-MiniLM-L6-v2
    Dimension: 384
    Distance: cosine (normalized vectors)

    Why this model:
    - 100% local, no API keys or network required after first download (~80MB)
    - Fast CPU inference: ~47 chunks embedded in <2 seconds
    - Well-tested on technical documentation and FAQ-style corpora
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or cfg.EMBEDDING_MODEL
        self.device = device or "cpu"
        logger.info(f"Loading embedding model '{self.model_name}' on {self.device}...")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension (384 for all-MiniLM-L6-v2)."""
        return self._model.get_embedding_dimension()

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Embed a list of text strings in batches.
        Returns standard Python float lists (not numpy arrays) for ChromaDB compatibility.
        """
        if not texts:
            return []
        bs = batch_size or cfg.EMBEDDING_BATCH_SIZE
        embeddings = self._model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Convert numpy array rows to plain Python float lists
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.
        Called at retrieval time for every user question.
        """
        return self.embed_texts([query])[0]
