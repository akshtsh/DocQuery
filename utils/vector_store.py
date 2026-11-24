"""Local Sentence-Transformers embedding and ChromaDB vector store utilities."""

import hashlib
import os
from typing import Any, Dict, List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "docquery_active_pdf"
TOP_K = 4


class LocalSentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible local wrapper around SentenceTransformer."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        try:
            # Prefer cached local weights first so offline runs never stall
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized embeddings for a list of document chunks."""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generate a normalized embedding for a single query string."""
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


_EMBEDDING_INSTANCE: Optional[LocalSentenceTransformerEmbeddings] = None


def get_embedding_model() -> LocalSentenceTransformerEmbeddings:
    """Return a cached singleton instance of the local embedding model."""
    global _EMBEDDING_INSTANCE
    if _EMBEDDING_INSTANCE is None:
        _EMBEDDING_INSTANCE = LocalSentenceTransformerEmbeddings(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_INSTANCE


def compute_document_id(file_bytes: bytes, filename: str) -> str:
    """Compute a deterministic identifier for an uploaded PDF."""
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    return f"{filename}::{digest}"


def create_vector_store(
    chunks: List[Document],
    doc_id: str,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """Replace the active ChromaDB collection with embeddings for the current PDF.

    Clears any previously stored collection so only the currently uploaded
    PDF is searched, then computes and persists embeddings once.
    """
    os.makedirs(persist_directory, exist_ok=True)
    embeddings = get_embedding_model()

    # Clear any previously indexed PDF from the active collection
    client = chromadb.PersistentClient(path=persist_directory)
    existing_collections = [col.name for col in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(name=COLLECTION_NAME)

    tagged_chunks: List[Document] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata["doc_id"] = doc_id
        tagged_chunks.append(
            Document(page_content=chunk.page_content, metadata=metadata)
        )

    vector_store = Chroma.from_documents(
        documents=tagged_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )
    return vector_store


def load_vector_store(persist_directory: str = CHROMA_PERSIST_DIR) -> Chroma:
    """Load the persisted ChromaDB collection without regenerating embeddings."""
    embeddings = get_embedding_model()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def search_chunks(
    vector_store: Chroma,
    query: str,
    k: int = TOP_K,
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve the top-k relevant chunks and preserve page metadata."""
    search_kwargs: Dict[str, Any] = {"k": k}
    if doc_id:
        search_kwargs["filter"] = {"doc_id": doc_id}

    try:
        docs = vector_store.similarity_search(query, **search_kwargs)
    except chromadb.errors.NotFoundError:
        persist_dir = getattr(vector_store, "_persist_directory", None) or CHROMA_PERSIST_DIR
        refreshed_store = load_vector_store(persist_directory=persist_dir)
        docs = refreshed_store.similarity_search(query, **search_kwargs)

    if not docs:
        return {
            "formatted_text": "No relevant passages were found in the uploaded PDF.",
            "sources": [],
        }

    formatted_parts: List[str] = []
    sources: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        page_num = doc.metadata.get("page_number")
        if page_num is None:
            page_num = int(doc.metadata.get("page", 0)) + 1
        source_file = doc.metadata.get("source_file", "Uploaded PDF")
        text = doc.page_content.strip()

        formatted_parts.append(f"[Chunk {idx} | Page {page_num}]\n{text}")
        sources.append(
            {
                "page": page_num,
                "source_file": source_file,
                "snippet": text,
            }
        )

    return {
        "formatted_text": "\n\n---\n\n".join(formatted_parts),
        "sources": sources,
    }
