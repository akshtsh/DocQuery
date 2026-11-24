"""PDF extraction and text chunking utilities for DocQuery."""

import os
import tempfile
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def process_pdf(file_bytes: bytes, filename: str) -> Tuple[List[Document], int]:
    """Extract text from PDF bytes using PyPDFLoader and split into chunks.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.
        filename: Original filename of the uploaded PDF.

    Returns:
        A tuple of (chunk_documents, total_pages).

    Raises:
        ValueError: If the file is empty, corrupted, not a valid PDF,
            or contains no extractable text.
    """
    if not file_bytes:
        raise ValueError("The uploaded PDF file is empty.")

    if not file_bytes[:1024].lstrip().startswith(b"%PDF"):
        raise ValueError("The uploaded file does not appear to be a valid PDF.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
    except Exception as exc:
        raise ValueError(f"Could not read the PDF file: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not pages:
        raise ValueError("The uploaded PDF has no readable pages.")

    # Retain 1-indexed page numbers and source filename in metadata
    non_empty_pages: List[Document] = []
    for idx, page_doc in enumerate(pages):
        raw_page = page_doc.metadata.get("page", idx)
        page_number = int(raw_page) + 1
        page_doc.metadata["page_number"] = page_number
        page_doc.metadata["source_file"] = filename
        if page_doc.page_content and page_doc.page_content.strip():
            non_empty_pages.append(page_doc)

    if not non_empty_pages:
        raise ValueError(
            "No extractable text was found in this PDF. "
            "It may be empty or consist only of scanned images."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(non_empty_pages)

    if not chunks:
        raise ValueError("Failed to split the PDF text into chunks.")

    for chunk_idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_idx

    return chunks, len(pages)
