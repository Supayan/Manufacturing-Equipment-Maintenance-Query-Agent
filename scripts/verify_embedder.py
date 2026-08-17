from app.ingest.embedder import MiniLMEmbedder
from app.config import settings
import numpy as np

embedder = MiniLMEmbedder(settings)

print(embedder.dim)

# max_seq_len = embedder._model.max_seq_length
# print(max_seq_len)
# vec=embedder.embed_query("test")
# print(len(vec))
# print(vec)
# norm = np.linalg.norm(vec)
# print(norm)

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

coolant = embedder.embed_query("coolant concentration")
refractometer = embedder.embed_query("refractometer coolant mix ratio")
spindle = embedder.embed_query("spindle drive belt replacement")
print(len(coolant))
print(len(refractometer))
print(len(spindle))
sim_related = cosine(coolant, refractometer)
sim_coolant_spindle = cosine(coolant, spindle)
sim_refractometer_spindle = cosine(refractometer, spindle)
print(sim_related)
print(sim_coolant_spindle)
print(sim_refractometer_spindle)

""" doc = embedder.embed_documents([])
# print(doc)
chunks = [f"maintenance log entry {i}: inspection" for i in range (100)]
vectors = embedder.embed_documents(chunks)
print(len(vectors))
print(len(vectors[0]))


load_lines = [
        chunks for chunks in chunks if "50" in chunks
    ]
print(len(load_lines)) """