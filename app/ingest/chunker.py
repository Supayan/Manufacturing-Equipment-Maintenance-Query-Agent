from dataclasses import dataclass
import pymupdf as pmp


@dataclass
class Chunk:
    chunk_id:int
    text:str
    doc_id:str
    doc_title:str
    page_start:int
    page_end:int
    chunk_index:int
    
def chunk_pdf(pdf_path,doc_id,doc_title,tokenizer, chunk_tokens, overlap_tokens, min_chunk_tokens, max_pages=None)-> list[Chunk]:
    doc = pmp.open(pdf_path)
    
    num_pages = 1
    if max_pages is not None:
        num_pages = min(max_pages, len(doc))
    else:
        num_pages = len(doc)

    token_id = []
    token_pages = []
    page_text = ""
    for pages in range(num_pages):
        page = doc[pages]
        page_text = page.get_text()
        tokens = tokenizer.encode(page_text, add_special_tokens=False)
        token_id.extend(tokens)
        token_pages.extend([pages]*len(tokens))
    chunks = []
    chunk_index = 1
    for i in range(0, len(token_id), chunk_tokens-overlap_tokens):
        last_index = min(i+ chunk_tokens, len(token_id))
        
        if last_index == len(token_id) and last_index -i < min_chunk_tokens and chunks:
            i= max(0, last_index-chunk_tokens)
            last_index = len(token_id)


        if last_index - i <min_chunk_tokens:
            continue
        
        chunk_token_id = token_id[i:last_index]
        chunk_text = tokenizer.decode(chunk_token_id, skip_special_tokens=False)
        page_start = token_pages[i] + 1
        page_end = token_pages[last_index-1] + 1

        chunks.append(Chunk(
            chunk_id = chunk_index,
            text=chunk_text,
            doc_id= doc_id,
            doc_title=doc_title,
            page_start=page_start,
            page_end=page_end,
            chunk_index=chunk_index,
        ))
        chunk_index+=1
    doc.close()
    return chunks