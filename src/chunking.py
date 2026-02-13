from nltk.tokenize import sent_tokenize

def semantic_chunk_text(text, max_tokens=120):
    sentences = sent_tokenize(text)
    chunks, current_chunk, tokens = [], [], 0

    for sentence in sentences:
        token_count = len(sentence.split())

        if tokens + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, tokens = [], 0

        current_chunk.append(sentence)
        tokens += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks