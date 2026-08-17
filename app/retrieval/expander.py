"""Retrieval-time neighbour expansion.

A section that exceeds the chunk token budget is split by the chunker into
several consecutive chunks that all carry the same heading. After the top-k
search, pull each hit's neighbouring chunks (chunk_index +/- window) from the
store and stitch the ones that belong to the same section back into a single
context block, so the LLM always sees complete procedures instead of half of
the steps.
"""

from app.storage.vector_store import Hit, chunk_uid


def _same_section(base_meta: dict, cand_meta: dict) -> bool:
    if base_meta.get("doc_id") != cand_meta.get("doc_id"):
        return False
    heading = base_meta.get("heading") or ""
    if heading:
        return (cand_meta.get("heading") or "") == heading
    # No heading to compare: accept only overlapping or adjacent pages.
    base = (int(base_meta.get("page_start", 0)), int(base_meta.get("page_end", 0)))
    cand = (int(cand_meta.get("page_start", 0)), int(cand_meta.get("page_end", 0)))
    return cand[0] <= base[1] + 1 and cand[1] >= base[0] - 1


def _strip_heading_lines(text: str, heading: str) -> str:
    """Drop the contextualize() heading prefix from a continuation chunk so
    the heading is not repeated mid-block."""
    if not heading:
        return text
    lines = text.split("\n")
    for part in heading.split(" > "):
        if lines and lines[0].strip() == part.strip():
            lines.pop(0)
    return "\n".join(lines)


def _build_block(doc_id: str, run: list[int], selected: dict) -> tuple[int, Hit]:
    members = [selected[(doc_id, i)] for i in run]
    heading = members[0]["meta"].get("heading") or ""
    texts = [members[0]["text"]]
    texts += [_strip_heading_lines(m["text"], heading) for m in members[1:]]
    meta = dict(members[0]["meta"])
    meta["page_start"] = min(int(m["meta"].get("page_start", 0)) for m in members)
    meta["page_end"] = max(int(m["meta"].get("page_end", 0)) for m in members)
    meta["stitched_chunks"] = len(members)
    chunk_id = chunk_uid(doc_id, run[0])
    if len(run) > 1:
        chunk_id += f"-{run[-1]:04d}"
    return (
        min(m["rank"] for m in members),
        Hit(
            chunk_id=chunk_id,
            text="\n".join(texts),
            metadata=meta,
            score=max(m["score"] for m in members),
        ),
    )


def expand_hits(hits: list[Hit], store, window: int = 1) -> list[Hit]:
    """Stitch chunks that were split out of the same section back together.

    For each hit, fetch chunk_index +/- window from the store and keep the
    neighbours whose heading matches the hit's heading (same section).
    Consecutive selected chunks are merged into one block in document order,
    deduplicated across hits, and the result is returned in the original
    retrieval-rank order. Hits without chunk metadata pass through unchanged.
    """
    if window <= 0 or not hits:
        return hits

    passthrough: list[tuple[int, Hit]] = []
    selected: dict[tuple[str, int], dict] = {}

    for rank, hit in enumerate(hits):
        meta = hit.metadata or {}
        doc_id = meta.get("doc_id")
        idx = meta.get("chunk_index")
        if doc_id is None or idx is None:
            passthrough.append((rank, hit))
            continue
        idx = int(idx)
        entry = selected.setdefault(
            (doc_id, idx),
            {"text": hit.text, "meta": meta, "score": hit.score, "rank": rank},
        )
        entry["score"] = max(entry["score"], hit.score)
        entry["rank"] = min(entry["rank"], rank)

        wanted = [
            i
            for i in range(idx - window, idx + window + 1)
            if i != idx and i >= 1 and (doc_id, i) not in selected
        ]
        if not wanted:
            continue
        for neighbour in store.get([chunk_uid(doc_id, i) for i in wanted]):
            n_meta = neighbour.metadata or {}
            n_idx = n_meta.get("chunk_index")
            if n_idx is None or not _same_section(meta, n_meta):
                continue
            selected.setdefault(
                (doc_id, int(n_idx)),
                {"text": neighbour.text, "meta": n_meta, "score": hit.score, "rank": rank},
            )

    # Merge consecutive chunk indexes of the same section into one block.
    blocks: list[tuple[int, Hit]] = list(passthrough)
    by_doc: dict[str, list[int]] = {}
    for doc_id, idx in selected:
        by_doc.setdefault(doc_id, []).append(idx)

    for doc_id, idxs in by_doc.items():
        idxs.sort()
        run: list[int] = []
        for idx in idxs:
            breaks_run = run and (
                idx != run[-1] + 1
                or (selected[(doc_id, idx)]["meta"].get("heading") or "")
                != (selected[(doc_id, run[-1])]["meta"].get("heading") or "")
            )
            if breaks_run:
                blocks.append(_build_block(doc_id, run, selected))
                run = []
            run.append(idx)
        if run:
            blocks.append(_build_block(doc_id, run, selected))

    blocks.sort(key=lambda pair: pair[0])
    return [hit for _, hit in blocks]
