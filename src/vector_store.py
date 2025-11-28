# src/vector_store.py

from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

# Directory to store Chroma DB on disk
DB_DIR = Path("data/chroma")
DB_DIR.mkdir(parents=True, exist_ok=True)

# Create a persistent Chroma client
client = chromadb.PersistentClient(path=str(DB_DIR))

# Use a small, fast sentence-transformer model for embeddings
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# One collection for now – can add more later
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_func,
)


def add_document(
    doc_id: str,
    chunks: List[str],
    metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Add chunks of a document into the vector DB.
    Each chunk gets an id like f"{doc_id}_{i}".
    """
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

    metadatas: List[Dict[str, Any]] = []
    for i in range(len(chunks)):
        base_meta: Dict[str, Any] = {
            "doc_id": doc_id,
            "chunk_index": i,
        }
        if metadata:
            base_meta.update(metadata)
        metadatas.append(base_meta)

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )


def query_document(
    query_text: str,
    n_results: int = 5,
    doc_id: str | None = None,
) -> Dict[str, Any]:
    """
    Query the vector store and get the most relevant chunks.
    If doc_id is provided, only search within that document.
    """
    where: Dict[str, Any] | None = None
    if doc_id is not None:
        where = {"doc_id": doc_id}

    result = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
    )
    return result


def list_documents() -> List[Dict[str, Any]]:
    """
    Return a list of documents present in the collection, with:
    - doc_id
    - source (filename)
    - num_chunks
    """
    data = collection.get()  # all points (ok for small projects)
    metadatas = data.get("metadatas", [])

    doc_stats: Dict[str, Dict[str, Any]] = {}

    for meta in metadatas:
        doc_id = meta.get("doc_id", "unknown")
        source = meta.get("source", "unknown.pdf")

        if doc_id not in doc_stats:
            doc_stats[doc_id] = {
                "doc_id": doc_id,
                "source": source,
                "num_chunks": 0,
            }

        doc_stats[doc_id]["num_chunks"] += 1

    return list(doc_stats.values())
