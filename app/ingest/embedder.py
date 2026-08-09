from typing import Protocol
from sentence_transformers import SentenceTransformer


class EmbeddingProvider(Protocol):
    @property
    def dim(self) -> int: ...
    @property
    def model_id(self) -> str: ...
    @property
    def tokenizer(self): ...
    @property
    def max_tokens(self) -> int:...
    def embed_documents(self, text: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...
class MiniLMEmbedder:
    def __init__(self, settings):
        self.model_name = settings.ingest.embedding.model
        self.device = settings.ingest.embedding.device
        self.cache_dir = settings.ingest.embedding.cache_dir
        self.batch_size = settings.ingest.embedding.batch_size
        self.normalize_embedding = settings.ingest.embedding.normalize
        self.query_prefix = settings.ingest.embedding.query_prefix
        self.document_prefix = settings.ingest.embedding.document_prefix
        # self.max_tokens = settings.generations.max_tokens

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=self.cache_dir,
        )
        # self._model_id = self.model_name
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self.model_name

    @property
    def tokenizer(self):
        return self._model.tokenizer
    
    @property
    def max_tokens(self)->int:
        return self._model.max_seq_length

    def embed_documents(self, text: list[str]) -> list[list[float]]:
        prefixed = [f"{self.document_prefix}{t}" for t in text]
        vectors = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embedding,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self._model.encode(
            f"{self.query_prefix}{query}",
            normalize_embeddings=self.normalize_embedding,
            convert_to_numpy=True,
        )
        return vector.tolist()
