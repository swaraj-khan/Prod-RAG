def rrf_fusion(bm25_docs, vector_docs, k=60):
    scores = {}
    all_docs = bm25_docs + vector_docs

    for rank, doc in enumerate(bm25_docs):
        scores[id(doc)] = 1 / (k + rank)

    for rank, doc in enumerate(vector_docs):
        scores[id(doc)] = scores.get(id(doc), 0) + 1 / (k + rank)

    ranked = sorted(
        all_docs,
        key=lambda d: scores.get(id(d), 0),
        reverse=True
    )

    return ranked[:10]