from dataclasses import dataclass
from typing import Protocol
import chromadb


@dataclass
class Hit:
    chunk_id: str
    text: str
    metadata: dict
    score: float


class VectorStoreProvider(Protocol):
    def upsert(self, ids, embeddings, documents, metadatas): ...
    def query(self, embedding, k, filters=None) -> list[Hit]: ...
    def count(self) -> int: ...
    def reset(self): ...


class ChromaVectorStore:
    def __init__(self, settings):
        self._settings = settings
        self.client = chromadb.PersistentClient(
            path=str(settings.app.paths.vector_store)
        )
        self.collections = self._get_or_create_collection()
        self.collection_name = settings.app.vector_store.collection

    def _get_or_create_collection(self):
        collections = self.client.get_or_create_collection(
            name=self._settings.app.vector_store.collection,
            metadata={"hnsw:space": self._settings.app.vector_store.distance,
                      "hnsw:search_ef": self._settings.app.vector_store.search_ef,

                      },
        )
        if self._settings.app.vector_store.distance != collections.metadata.get(
            "hnsw:space"
        ):
            raise RuntimeError(
                "Vector store distance mismatch: "
                f"expected {self._settings.app.vector_store.distance}, "
                f"found {collections.metadata.get('hnsw:space')}."
            )
        return collections

    def upsert(self, ids, embeddings, documents, metadatas):
        self.collections.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def count(self):
        return self.collections.count()

    def reset(self):
        self.client.delete_collection(self.collection_name)
        self.collections = self._get_or_create_collection()

    def query(self, embedding, k, filters=None) -> list[Hit]:
        result = self.collections.query(
            query_embeddings=[embedding],
            n_results=k,
            where=filters,
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        hits = []
        for chunk_id, text, metadata, score in zip(
            ids, documents, metadatas, distances
        ):
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    score=1 - score,
                )
            )
        return hits
