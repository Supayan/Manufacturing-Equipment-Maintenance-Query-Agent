from app.ingest.embedder import MiniLMEmbedder
from app.ingest.chunker import chunk_pdf
from app.config import settings
from pathlib import Path
import pprint

embedder = MiniLMEmbedder(settings)

tokenizer = embedder.tokenizer
chunks = chunk_pdf(
    "corpus/inbox/2023_mill_operator_man_hass.pdf",
    "Hass_doc_2023",
    "2023_mill_operator_man_hass",
    tokenizer,
    settings.ingest.chunk_tokens,
    settings.ingest.overlap_tokens,
    settings.ingest.min_chunk_tokens,
)
# print(chunks)
output_path = Path("chunks_output.txt")
with output_path.open("w", encoding="utf-8") as f:
    f.write(pprint.pformat(chunks))


