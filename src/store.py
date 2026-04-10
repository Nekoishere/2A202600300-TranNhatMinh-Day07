from __future__ import annotations

from typing import Any, Callable

from .chunking import FixedSizeChunker, _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        metadata = {"doc_id": doc.id, **doc.metadata}  # doc.metadata wins (preserves pre-set doc_id)
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored = [(record, _dot(query_embedding, record["embedding"])) for record in records]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{**record, "score": score} for record, score in scored[:top_k]]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            chunks = self._chunker.chunk(doc.content)
            for i, chunk_text in enumerate(chunks):
                chunk_doc = Document(
                    id=f"{doc.id}_chunk_{i}",
                    content=chunk_text,
                    metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": i},
                )
                if self._use_chroma:
                    embedding = self._embedding_fn(chunk_text)
                    self._collection.add(
                        ids=[chunk_doc.id],
                        documents=[chunk_text],
                        embeddings=[embedding],
                        metadatas=[chunk_doc.metadata],
                    )
                else:
                    self._store.append(self._make_record(chunk_doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
            return [
                {"id": id_, "content": doc, "metadata": meta}
                for id_, doc, meta in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                )
            ]
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter:
            filtered = [
                record for record in self._store
                if all(record["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        else:
            filtered = self._store
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        original_len = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        return len(self._store) < original_len
