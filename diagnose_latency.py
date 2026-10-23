"""
diagnose_latency.py
Latency and model diagnostic script for Enso Group HR Policy Assistant.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

from ingest import get_clean_key
from rag import get_vectorstore, get_chat_model

api_key = get_clean_key()
if not api_key:
    print("[ERROR] No valid GEMINI_API_KEY or GOOGLE_API_KEY found in environment or .env.")
    exit(1)


def redact(text: str) -> str:
    if api_key and api_key in text:
        return text.replace(api_key, "[REDACTED_API_KEY]")
    return text


print("=" * 60)
print("1. DIAGNOSING VECTORSTORE RETRIEVAL LATENCY")
print("=" * 60)
t0 = time.perf_counter()
try:
    vs = get_vectorstore(api_key=api_key)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke("How many days of Privilege Leave (PL) does a permanent employee get per year?")
    t_retrieval = time.perf_counter() - t0
    print(f"[SUCCESS] Retrieval completed in {t_retrieval:.2f} seconds.")
    print(f"Retrieved {len(docs)} documents.")
except Exception as e:
    t_retrieval = time.perf_counter() - t0
    print(f"[FAIL] Retrieval failed after {t_retrieval:.2f} seconds.")
    print(f"Error: {redact(str(e))}")

print("\n" + "=" * 60)
print("2. DIAGNOSING CHAT MODEL INVOCATION LATENCY")
print("=" * 60)
t0 = time.perf_counter()
configured_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash")
print(f"Configured GEMINI_CHAT_MODEL: '{configured_model}'")
try:
    chat_model = get_chat_model(api_key=api_key)
    resp = chat_model.invoke("Reply with OK")
    t_chat = time.perf_counter() - t0
    print(f"[SUCCESS] Chat model responded in {t_chat:.2f} seconds.")
    resp_text = resp.content
    if isinstance(resp_text, list):
        resp_text = " ".join([p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in resp_text])
    print(f"Response: {resp_text}")
except Exception as e:
    t_chat = time.perf_counter() - t0
    print(f"[FAIL] Chat model failed after {t_chat:.2f} seconds.")
    print(f"Error: {redact(str(e))}")

print("\n" + "=" * 60)
print("3. LISTING AVAILABLE GEMINI MODELS")
print("=" * 60)
t0 = time.perf_counter()
try:
    from google import genai
    client = genai.Client(api_key=api_key)
    models = list(client.models.list())
    t_list = time.perf_counter() - t0
    print(f"[SUCCESS] Model list retrieved in {t_list:.2f} seconds.")
    
    filtered_models = [
        m.name for m in models 
        if "flash" in (m.name or "").lower() or "embedding" in (m.name or "").lower()
    ]
    print(f"Found {len(filtered_models)} models matching 'flash' or 'embedding':")
    for m_name in sorted(filtered_models):
        print(f"  - {m_name}")
except Exception as e:
    t_list = time.perf_counter() - t0
    print(f"[FAIL] Model list failed after {t_list:.2f} seconds.")
    print(f"Error: {redact(str(e))}")
print("=" * 60)
