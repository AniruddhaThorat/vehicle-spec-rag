from src.pdf_reader import read_pdf
from src.chunking import semantic_chunk_text
from src.embedding import create_faiss_index
from src.retrieval import retrieve
from src.extractor import extract_specs

def run_pipeline(pdf_path, query):

    text = read_pdf(pdf_path)
    chunks = semantic_chunk_text(text)

    index, _ = create_faiss_index(chunks)

    context_chunks = retrieve(query, chunks, index)
    context = " ".join(context_chunks)

    result = extract_specs(context, query)

    return result


def batch_queries(pdf_path, queries):

    text = read_pdf(pdf_path)
    chunks = semantic_chunk_text(text)
    index, _ = create_faiss_index(chunks)

    all_results = []

    for q in queries:
        context_chunks = retrieve(q, chunks, index)
        context = " ".join(context_chunks)

        result = extract_specs(context, q)

        all_results.append({
            "query": q,
            "result": result
        })

    return all_results