"""
ingest.py
Enso Group HR Policy Manual — Resumable, Rate-Limited ChromaDB Ingestion Pipeline
Features: Stable Chunk IDs, Deduplication, Safe Resumption, Gemini Rate-Limiting (RPM Guard),
Exponential Backoff with Jitter for 429 Errors, and Comprehensive Validation.
"""

import os
import sys
import time
import random
import hashlib
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


DEFAULT_PDF_PATH = "data/hr_manual.pdf"
DEFAULT_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "enso_hr_policy"
TOTAL_EXPECTED_CHUNKS = 279


def get_embedding_model(api_key: str = None):
    """Initializes GoogleGenerativeAIEmbeddings using GEMINI_API_KEY or GOOGLE_API_KEY."""
    resolved_key = (
        api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not resolved_key or not resolved_key.strip():
        raise ValueError(
            "Missing Gemini API Key. Please enter your Gemini API Key in the Streamlit application sidebar "
            "or set GEMINI_API_KEY in your local .env file."
        )

    clean_key = resolved_key.strip()
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001").strip()
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    try:
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=clean_key,
        )
    except Exception as e:
        err_str = str(e)
        if clean_key in err_str:
            err_str = err_str.replace(clean_key, "[REDACTED_API_KEY]")
        raise RuntimeError(f"Failed to initialize Gemini embedding model '{model_name}': {err_str}") from None


def load_and_split_pdf(pdf_path: str = DEFAULT_PDF_PATH, chunk_size: int = 800, chunk_overlap: int = 120):
    """
    Loads the PDF using PyPDFLoader and splits it into chunks (~800 chars, 120 overlap).
    Preserves original 1-based page numbers in chunk metadata.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Authoritative PDF not found at '{pdf_path}'. "
            "Please ensure data/hr_manual.pdf exists before running ingestion."
        )

    loader = PyPDFLoader(str(path))
    pages = loader.load()
    if not pages:
        raise ValueError(f"No pages could be extracted from '{pdf_path}'. File may be empty or corrupted.")

    # Standardize metadata: 1-based page number and uniform source path
    for p in pages:
        raw_page = p.metadata.get("page", 0)
        p.metadata["page"] = int(raw_page) + 1
        p.metadata["source"] = str(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    return chunks


def build_vectorstore(
    pdf_path: str = DEFAULT_PDF_PATH,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    api_key: str = None,
    force_reindex: bool = False,
    progress_callback=None,
    batch_size: int = 5,
    delay_between_batches: float = 6.2,  # ~48 items/min to stay comfortably below 100 RPM
):
    """
    Creates or safely resumes ChromaDB vector store indexing.
    - Inspects existing chunks to avoid re-embedding what is already in ChromaDB.
    - Embeds missing chunks with a conservative rate limiter (stays under ~50 items/min).
    - Features exponential backoff with jitter on 429 / RESOURCE_EXHAUSTED errors.
    - Persists every batch immediately so progress is never lost.
    """
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = get_embedding_model(api_key=api_key)
    clean_key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()

    # Initialize Chroma vectorstore
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    chunks = load_and_split_pdf(pdf_path=pdf_path)
    total_expected = len(chunks)

    # Handle force_reindex
    if force_reindex:
        print("[*] Force re-indexing requested: Clearing existing ChromaDB collection...")
        vectorstore.delete_collection()
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        existing_hashes = set()
        existing_ids = set()
        existing_count = 0
    else:
        # Inspect existing records in ChromaDB to safely resume
        existing_data = vectorstore._collection.get(include=["metadatas", "documents"])
        existing_ids = set(existing_data.get("ids") or [])
        existing_docs = existing_data.get("documents") or []
        existing_hashes = {
            hashlib.sha256(doc.encode("utf-8")).hexdigest()
            for doc in existing_docs
        }
        existing_count = len(existing_ids)
        print(f"[*] ChromaDB currently has {existing_count} chunks indexed.")

    # If all expected chunks are already present, verify and return
    if not force_reindex and existing_count >= total_expected and len(existing_hashes) >= total_expected:
        print(f"[✓] All {total_expected} chunks are already indexed in ChromaDB. Skipping embedding.")
        if progress_callback:
            progress_callback(total_expected, total_expected, f"All {total_expected} chunks already indexed.")
        return vectorstore, total_expected

    # Identify precisely which chunks are missing
    chunks_to_index = []
    ids_to_index = []
    for idx, chunk in enumerate(chunks):
        chunk_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        stable_id = f"enso_hr_{idx:03d}_{chunk_hash[:12]}"
        if (chunk_hash in existing_hashes) or (stable_id in existing_ids):
            continue
        chunks_to_index.append((idx, chunk))
        ids_to_index.append(stable_id)

    missing_count = len(chunks_to_index)
    print(f"[*] Resuming indexing: {existing_count} chunks already saved, {missing_count} chunks remaining.")
    if progress_callback:
        progress_callback(
            existing_count,
            total_expected,
            f"Resuming indexing ({existing_count}/{total_expected} already saved, {missing_count} remaining)...",
        )

    if missing_count == 0:
        return vectorstore, existing_count

    # Process missing chunks in rate-limited batches with retry logic
    max_retries = 6
    for b_start in range(0, missing_count, batch_size):
        b_end = min(b_start + batch_size, missing_count)
        batch_items = chunks_to_index[b_start:b_end]
        batch_chunks = [item[1] for item in batch_items]
        batch_ids = ids_to_index[b_start:b_end]
        batch_texts = [c.page_content for c in batch_chunks]
        batch_metas = [c.metadata for c in batch_chunks]

        # Embed with bounded exponential backoff on 429
        embeddings_list = None
        for attempt in range(max_retries):
            try:
                embeddings_list = embeddings.embed_documents(batch_texts)
                break
            except Exception as e:
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "quota" in err_str.lower()
                    or "rate limit" in err_str.lower()
                )
                if is_rate_limit and attempt < max_retries - 1:
                    # Exponential backoff with jitter: 15s -> 30s -> 60s
                    backoff = min(75.0, (2 ** attempt) * 15.0 + random.uniform(1.0, 5.0))
                    status_msg = (
                        f"Gemini quota rate limit reached (429). Pausing for {backoff:.0f}s before retry "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    print(f"[*] {status_msg}")
                    if progress_callback:
                        progress_callback(existing_count + b_start, total_expected, status_msg)
                    time.sleep(backoff)
                else:
                    safe_err = err_str.replace(clean_key, "[REDACTED_API_KEY]") if clean_key in err_str else err_str
                    raise RuntimeError(
                        f"Gemini embedding quota limit: {safe_err}. "
                        f"Successfully persisted {existing_count + b_start}/{total_expected} chunks in ChromaDB. "
                        f"You can resume indexing safely without losing progress."
                    ) from None

        # Upsert the batch into ChromaDB with stable IDs
        vectorstore._collection.upsert(
            ids=batch_ids,
            embeddings=embeddings_list,
            documents=batch_texts,
            metadatas=batch_metas,
        )

        current_total = existing_count + b_end
        page_ref = batch_chunks[-1].metadata.get("page", "?")
        print(f"    Indexed {current_total}/{total_expected} chunks (Page {page_ref})")

        if progress_callback:
            progress_callback(
                current_total,
                total_expected,
                f"Indexed {current_total}/{total_expected} chunks (Page {page_ref})...",
            )

        # Conservative rate limiter delay between batches
        if b_end < missing_count:
            time.sleep(delay_between_batches)

    final_count = vectorstore._collection.count()
    print(f"[✓] Ingestion verified: {final_count}/{total_expected} chunks in ChromaDB.")
    return vectorstore, final_count


def check_index_status(persist_dir: str = DEFAULT_PERSIST_DIR):
    """Checks if the Chroma index exists and returns chunk count without loading embeddings."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()
        for col in collections:
            if col.name == COLLECTION_NAME:
                return col.count()
        return 0
    except Exception:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest HR Policy Manual into ChromaDB.")
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH, help="Path to HR manual PDF")
    parser.add_argument("--dir", default=DEFAULT_PERSIST_DIR, help="ChromaDB persistence directory")
    parser.add_argument("--force", action="store_true", help="Force re-indexing even if data exists")
    parser.add_argument("--api-key", default=None, help="Explicit Gemini API Key")
    args = parser.parse_args()

    try:
        vectorstore, count = build_vectorstore(
            pdf_path=args.pdf,
            persist_dir=args.dir,
            api_key=args.api_key,
            force_reindex=args.force,
        )
        print(f"[✓] Success: ChromaDB collection '{COLLECTION_NAME}' contains {count} chunks.")
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)
