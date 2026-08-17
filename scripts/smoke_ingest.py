import argparse

from app.config import settings
from app.ingest.docling_chunker import chunk_pdf
from app.ingest.embedder import MiniLMEmbedder
from app.storage.vector_store import ChromaVectorStore, chunk_uid

import csv
import json

MANIFEST_PATH = settings.app.paths.corpus_inbox.parent / "manifest.csv"


def load_manifest_row(doc_id: str) -> dict:
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as man:
        rows = list(csv.DictReader(man))
    for row in rows:
        if row["doc_id"] == doc_id:
            return row
    available = ", ".join(row["doc_id"] for row in rows)
    raise ValueError(
        f"doc_id {doc_id!r} not found in {MANIFEST_PATH}. Available ids: {available}"
    )


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main():
    DOC_ID = "doc_haas_2023"
    RESET = True

    embedder = MiniLMEmbedder(settings)
    store = ChromaVectorStore(settings)

    if RESET == True:
        store.reset()

    row = load_manifest_row(DOC_ID)
    pdf_path = settings.app.paths.corpus_inbox / row["filename"]

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"No file at {pdf_path} for doc_id {DOC_ID!r} (manifest filename: {row['filename']!r})"
        )

    doc_title = pdf_path.stem
    max_tokens = embedder.max_tokens
    chunks = chunk_pdf(
        str(pdf_path),
        DOC_ID,
        doc_title,
        embedder.tokenizer,
        max_tokens,
        settings.ingest.min_chunk_tokens,
    )

    if not chunks:
        print(f"No chunks produced for {DOC_ID} ({pdf_path.name}) — nothing to upsert.")
        return

    ids = [chunk_uid(DOC_ID, chunk.chunk_index) for chunk in chunks]

    texts = [chunk.text for chunk in chunks]

    metadatas = [
        {
            "doc_id":DOC_ID,
            "doc_title": doc_title,
            "doc_type": row["doc_type"],
            "manufacturer": row["manufacturer"],
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chunk_index": chunk.chunk_index,
            "heading": chunk.heading,
            "category": chunk.category,
            "prov": json.dumps(chunk.prov),
        }
        for chunk in chunks
    ]
    
    vectors = embedder.embed_documents(texts)

    batch_size = settings.ingest.embedding.batch_size
    for batch_id, vector_batch, text_batch, meta_batch in zip(
        batched(ids, batch_size),
        batched(vectors, batch_size),
        batched(texts, batch_size),
        batched(metadatas, batch_size),
    ):
        store.upsert(ids=batch_id, embeddings= vector_batch, documents=text_batch, metadatas=meta_batch)
        
    print(f"doc_id: {DOC_ID}")
    print(f"file: {pdf_path.name}")
    print(f"doc_type: {row['doc_type']} | manufacturer: {row['manufacturer']}")
    print(f"chunks ingested: {len(chunks)}")
    print(f"page range: {chunks[0].page_start}-{chunks[-1].page_end}")
    print(f"vector store count: {store.count()}")
    
if __name__ == "__main__":
    main()