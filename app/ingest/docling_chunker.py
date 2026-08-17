from bisect import bisect_right
from dataclasses import dataclass
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.labels import DocItemLabel

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.types.doc.common.content_layer import ContentLayer

_TABLE_LABELS = {DocItemLabel.TABLE}
_LIST_LABELS = {DocItemLabel.LIST_ITEM}
_HEADING_LABELS = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}


@dataclass
class Chunk:
    chunk_id: int
    text: str
    doc_id: str
    doc_title: str
    page_start: int
    page_end: int
    chunk_index: int
    heading: str
    category: str
    prov: list[tuple[int, float, float, float, float, str]]

def _common_prefix(a,b):
    out= []
    for x,y in zip(a or[], b or []):
        if x!=y:
            break
        out.append(x)
    return out


def _join(first, second, delim= "\n"):
    headings= _common_prefix(first.meta.headings, second.meta.headings)
    if (first.meta.headings or second.meta.headings) and not headings:
        return None
    return DocChunk(
        text = delim.join([first.text, second.text]),
        meta = DocMeta(
          doc_items = [*first.meta.doc_items, *second.meta.doc_items],
          headings= headings or None,
          origin = first.meta.origin or second.meta.origin,  
        ),
    )
    
def merge_undersized(chunks, min_tokens, max_tokens, size_of):
    chunks = list(chunks)
    changed = True
    while changed:
        changed=False
        for i, chunk in enumerate(chunks):
            if size_of(chunk)>= min_tokens:
                continue
            pairs=[]
            if i>0:
                pairs.append((i-1,i))
            if i<len(chunks)-1:
                pairs.append((i,i+1))
            pairs.sort(key=lambda p:size_of(chunks[p[0] if p[1] == i else p[1]]))
            
            for lo, hi, in pairs:
                merged = _join(chunks[lo], chunks[hi])
                if merged is None or size_of(merged) > max_tokens:
                    continue
                chunks[lo:hi + 1] = [merged]
                changed=True
                break
            if changed:
                break
            
    return chunks
    
def _categorize(labels: set) -> str:
    if labels & _TABLE_LABELS:
        return "table"
    if labels & _HEADING_LABELS:
        return "heading"
    if labels & _LIST_LABELS:
        return "list"
    return "text"


def _build_converter(ocr_enabled: bool, ocr_dpi: int) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        mode=TableFormerMode.ACCURATE
    )
    pipeline_options.do_ocr = ocr_enabled
    if ocr_enabled:
        pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
def _first_prov(node, doc):
    """First ProvenanceItem of a node or any of its descendants (BFS)."""
    queue = [node]
    while queue:
        item = queue.pop(0)
        prov = getattr(item, "prov", None)
        if prov:
            return prov[0]
        queue.extend(ref.resolve(doc) for ref in getattr(item, "children", []) or [])
    return None


def reorder_body_reading_order(dl_doc, col_frac=0.5, band_tol=6.0):
    """Re-sort top-level body items into geometric reading order.

    Haas-style pages are laid out in horizontal *bands*: each section heading
    in the left column starts a new band, and inside a band the text flows
    left column first, then right column. Docling instead serializes the whole
    left column before the right column, which pushes right-column steps under
    the *next* section's heading (e.g. the POWER UP step of "Machine Power-On"
    landing inside "Spindle Warm-Up"). Sorting by (page, band, column, top)
    restores the true order so the chunker attaches every item to the correct
    heading. Returns the number of items that moved.
    """
    children = list(dl_doc.body.children)

    infos = []
    for ref in children:
        item = ref.resolve(dl_doc)
        prov = _first_prov(item, dl_doc)
        page = dl_doc.pages.get(prov.page_no) if prov else None
        if page is None:
            infos.append(None)
            continue
        bbox = prov.bbox.to_top_left_origin(page.size.height)
        col = 1 if bbox.l >= page.size.width * col_frac else 0
        is_heading = getattr(item, "label", None) in _HEADING_LABELS
        infos.append((prov.page_no, col, bbox.t, is_heading))

    # A new band starts at each left-column section heading.
    band_starts: dict[int, list[float]] = {}
    for info in infos:
        if info and info[3] and info[1] == 0:
            band_starts.setdefault(info[0], []).append(info[2])
    for tops in band_starts.values():
        tops.sort()

    keys = []
    last_key = (-1, 0, 0, 0.0)
    for info in infos:
        if info is None:  # no geometry: stay glued to the previous item
            keys.append(last_key)
            continue
        page_no, col, top, _ = info
        band = bisect_right(band_starts.get(page_no, []), top + band_tol)
        last_key = (page_no, band, col, top)
        keys.append(last_key)

    order = sorted(range(len(children)), key=keys.__getitem__)
    moved = sum(1 for pos, i in enumerate(order) if i != pos)
    if moved:
        dl_doc.body.children = [children[i] for i in order]
    return moved

def chunk_pdf(
    pdf_path,
    doc_id,
    doc_title,
    tokenizer,
    chunk_tokens,
    min_chunk_tokens,
    ocr_enabled=False,
    ocr_dpi=300,
    max_pages=None,
    fix_reading_order=True,
) -> list[Chunk]:
    converter = _build_converter(ocr_enabled, ocr_dpi)
    convert_kwargs = {"max_num_pages": max_pages} if max_pages is not None else {}

    result = converter.convert(pdf_path, **convert_kwargs)

    dl_doc = result.document
    if dl_doc is None:
        raise RuntimeError(
            f"Docling failed to convert {pdf_path}: status={result.status}"
        )
        

    demoted=[]
    for item, _ in dl_doc.iterate_items(with_groups=True):
        if getattr(item, "label", None) in {
            DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER
        }:
            item.content_layer = ContentLayer.FURNITURE
            demoted.append((item.self_ref, getattr(item, "text","")[:60]))
    print(f"demoted {len(demoted)} furniture Items")

    if fix_reading_order:
        moved = reorder_body_reading_order(dl_doc)
        print(f"reading order: moved {moved} top-level items")

    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=chunk_tokens),
        merge_peers=True,
    )

    chunks = []
    chunk_index = 1
    
    _sizes={}
    def size_of(c):
        key = (tuple(c.meta.headings or ()), c.text)
        if key not in _sizes:   
            _sizes[key]= len(
                tokenizer.encode(chunker.contextualize(c), add_special_tokens=False)
            )
        return _sizes[key]    


    raw_chunks = merge_undersized(
        list(chunker.chunk(dl_doc)),
        min_tokens = min_chunk_tokens,
        max_tokens= chunk_tokens,
        size_of = size_of,
    )
    for raw_chunk in raw_chunks:
        text = chunker.contextualize(raw_chunk)
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        # if token_count < min_chunk_tokens:
        #     continue

        doc_items = raw_chunk.meta.doc_items
        labels = {item.label for item in doc_items}
        pages = {prov.page_no for item in doc_items for prov in item.prov}
        headings = raw_chunk.meta.headings or []

        prov_list = []
        for items in doc_items:
            for prov in items.prov:
                page_no = prov.page_no
                page_height = dl_doc.pages[page_no].size.height
                bbox= prov.bbox.to_top_left_origin(page_height)
                prov_list.append((page_no,bbox.l,bbox.t, bbox.r, bbox.b, bbox.coord_origin.value))

        chunks.append(
            Chunk(
                chunk_id=chunk_index,
                text=text,
                doc_id=doc_id,
                doc_title=doc_title,
                page_start=min(pages) if pages else 1,
                page_end=max(pages) if pages else 1,
                chunk_index=chunk_index,
                heading=" > ".join(headings),
                category=_categorize(labels),
                prov=prov_list,
            )
        )

        chunk_index += 1
    anomalies_check = anomalies(chunks, dl_doc)
    if anomalies_check:
        for a in anomalies_check:
            print("anomalies: ",a)
    return chunks

def compute_topmost_per_page(chunk, dl_doc) -> dict[int, float]:
    """For a given chunk, compute the topmost normalized coordinate per page
    return dict{page_No : topmost_value}"""
    topmost_value: dict[int, float] = {}
    for prov in chunk.prov:
        page_no, l, t, r,b,origin = prov
        
        if page_no not in topmost_value or t<topmost_value[page_no]:
            topmost_value[page_no]=t
            
    return topmost_value

def anomalies(chunks: list[Chunk], dl_doc)-> list[dict]:
    anomalies: list[dict] = []
    for i in range(len(chunks)-1):
        chunk_n = chunks[i]
        chunk_n1 = chunks[i+1]
        
        top_n = compute_topmost_per_page(chunk_n, dl_doc)
        top_n1 = compute_topmost_per_page(chunk_n1, dl_doc)
        
        shared_pages = set(top_n.keys())& set(top_n1.keys())
        for page in shared_pages:
            if top_n1[page] < top_n[page]:
                anomalies.append({
                    "chunk_n_index": i,
                    "chunk_n1_index": i+1,
                    "page": page,
                    "top_n": top_n[page],
                    "top_n1": top_n1[page],
                    "chunk_n_id": getattr(chunk_n, "chunk_id", None),
                    "chunk_n1_id": getattr(chunk_n1, "chunk_id", None),
                })
    return anomalies