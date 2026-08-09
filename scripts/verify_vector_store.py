from app.storage.vector_store import ChromaVectorStore
from app.config import settings
from app.ingest.embedder import MiniLMEmbedder

CHUNK_IDS = ["chunk1","chunk2","chunk3"]

CHUNK_TEXTS = [
    "Coolant concentration should be checked weekly with a refractometer.",
    "Spindle drive belt tension must be inspected every 500 operating hours.",
    "Hydraulic fluid should be replaced every 2000 hours or annually.",
]
CHUNK_METADATAS = [
    {"section": "coolant"},
    {"section": "spindle"},
    {"section": "hydraulics"},
]

embedder = MiniLMEmbedder(settings)
store = ChromaVectorStore(settings)
# seeding
vectors = embedder.embed_documents(CHUNK_TEXTS)
# print(vectors)
store.upsert(ids=CHUNK_IDS, embeddings= vectors, documents=CHUNK_TEXTS, metadatas= CHUNK_METADATAS)
print(store.count())

hits = store.query(vectors[1],k=2)
print(f"Id : {hits[0].chunk_id}; score: {hits[0].score}")

scores = [hit.score for hit in hits]
print(scores)
# persist
print(store)
count = store.count()
print(count)

store.reset()
print(store.count())

query= embedder.embed_query("hydraulics")
hits = store.query(query,k=2)
print(hits)