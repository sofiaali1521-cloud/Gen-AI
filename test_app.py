"""
test_app.py
Automated unit tests for Enso Group HR Policy Assistant
"""

import os
import pytest
from ingest import load_and_split_pdf, check_index_status, DEFAULT_PDF_PATH
from rag import answer_question


def test_pdf_exists():
    assert os.path.exists(DEFAULT_PDF_PATH), f"PDF should exist at {DEFAULT_PDF_PATH}"


def test_chunking_and_metadata():
    chunks = load_and_split_pdf(DEFAULT_PDF_PATH, chunk_size=800, chunk_overlap=120)
    assert len(chunks) > 100, f"Expected >100 chunks, got {len(chunks)}"
    
    # Check 1-based page metadata
    pages = [c.metadata.get("page") for c in chunks]
    assert all(p is not None and 1 <= p <= 79 for p in pages), "All page numbers must be 1-based between 1 and 79"
    assert min(pages) == 1, "First page should be 1"
    assert max(pages) == 79, "Last page should be 79"


def test_empty_query_handling():
    res = answer_question("", api_key="dummy_key")
    assert "valid question" in res["answer"].lower()
    assert res["is_found"] is False


def test_whitespace_query_handling():
    res = answer_question("    \n\t   ", api_key="dummy_key")
    assert "valid question" in res["answer"].lower()
    assert res["is_found"] is False


def test_missing_index_handling():
    # Points to non-existent chroma directory
    res = answer_question("Test question", api_key="dummy_key", persist_dir="non_existent_chroma_dir_test")
    assert "empty" in res["answer"].lower() or "not found" in res["answer"].lower()
