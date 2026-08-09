from urllib import response

from app.config import settings
from app.generation.providers import OllamaProvider
from app.generation.qa import answer_query, build_user_prompt, format_citations
from app.ingest.embedder import MiniLMEmbedder
from app.storage.vector_store import ChromaVectorStore

import time


def main():
    QUERY = "What is the machine power-on procedure?"
    k = settings.retrieval.rerank_top_n
    SHOW_CONTEXT = False
    
    embedder = MiniLMEmbedder(settings)
    data_store = ChromaVectorStore(settings)
    provider = OllamaProvider(settings)
    
    print(data_store.count())
    
    if data_store.count() == 0:
        print("Empty Collection, run smoke_ingest")
        raise SystemExit(1)
    
    if SHOW_CONTEXT:
        query_vector = embedder.embed_query(QUERY)
        hits= data_store.query(query_vector, k=k)
        print(build_user_prompt(QUERY, hits))
        return
    
    start = time.perf_counter()
    response, citations = answer_query(QUERY, embedder, data_store, provider, k=k)
    latency_ms = round((time.perf_counter()- start)*1000)
    print(f"Query: {QUERY}\n")
    print(f"{response.text}\n")
    
    for line in format_citations(citations):
        print(line)
    print(f"latency: {latency_ms}ms")
    print(f"model: {response.model}")


if __name__ == "__main__":
    main()