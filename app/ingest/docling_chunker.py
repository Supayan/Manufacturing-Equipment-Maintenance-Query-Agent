from dataclasses import dataclass
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.labels import DocItemLabel

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


def _categorize(labels: set) -> str:
    if labels & _TABLE_LABELS:
        return "table"
    if labels & labels <= _HEADING_LABELS:
        return "heading"
    if labels & _LIST_LABELS:
        return "list"
    return "text"


def _build_converter(ocr_enabled: bool, ocr_dpi: int) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.do_ocr = ocr_enabled
    if ocr_enabled:
        pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)

    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options = pipeline_options)
    })


def chunk_pdf(pdf_path,doc_id,doc_title,tokenizer, chunk_tokens, min_chunk_tokens, ocr_enabled= False,ocr_dpi= 300,max_pages=None)-> list[Chunk]:
    converter = _build_converter(ocr_enabled, ocr_dpi)
    convert_kwargs = {
        "max_num_pages": max_pages
    }if max_pages is not None else {}
    
    result = converter.convert(pdf_path, **convert_kwargs)

    dl_doc = result.document
    if dl_doc is None:
        raise RuntimeError(f"Docling failed to convert {pdf_path}: status={result.status}")
    
    chunker = HybridChunker(tokenizer= tokenizer, max_tokens= chunk_tokens, merge_peers= True)
    
    chunks =[]
    chunk_index = 1
    for raw_chunk in chunker.chunk(dl_doc):
        text = chunker.contextualize(raw_chunk)
        token_count = len(tokenizer.encode(text, add_special_tokens = False))
        if token_count < min_chunk_tokens:
            continue
        
        doc_items = raw_chunk.meta.doc_items
        labels = {item.label for item in doc_items}
        pages = {prov.page_no for item in doc_items for prov in item.prov}
        headings = raw_chunk.meta.headings or []
        
        chunks.append(Chunk(
            chunk_id = chunk_index,
            text= text,
            doc_id= doc_id,
            doc_title= doc_title,
            page_start= min(pages) if pages else 1,
            page_end= max(pages)if pages else 1,
            chunk_index= chunk_index,
            heading= " > ".join(headings),
            category=_categorize(labels),
        ))
        
        chunk_index+=1
    return chunks