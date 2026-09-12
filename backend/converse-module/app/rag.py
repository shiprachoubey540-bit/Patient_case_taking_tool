import json
import httpx
import numpy as np
from pathlib import Path

# We'll use the Ollama embedding model the user just pulled
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

_guidelines = []
_vectors = None

def get_embedding(text: str) -> np.ndarray:
    try:
        resp = httpx.post(OLLAMA_URL, json={
            "model": EMBED_MODEL,
            "prompt": text
        }, timeout=10.0)
        resp.raise_for_status()
        embedding = resp.json()["embedding"]
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"[RAG] Error getting embedding: {e}")
        return np.zeros(768, dtype=np.float32)  # fallback shape for nomic

def _load_and_embed():
    global _guidelines, _vectors
    if _vectors is not None:
        return
        
    guidelines_path = Path(__file__).parent / "guidelines.json"
    if not guidelines_path.exists():
        _guidelines = []
        _vectors = np.empty((0, 768))
        return

    with open(guidelines_path, "r", encoding="utf-8") as f:
        _guidelines = json.load(f)

    print(f"[RAG] Loading {len(_guidelines)} clinical guidelines into vector space...")
    embeds = []
    for g in _guidelines:
        # Embed the condition name and the protocol together so it matches user symptoms
        text = f"Patient symptom: {g['condition']}. Protocol: {g['protocol']}"
        embeds.append(get_embedding(text))
    
    if embeds:
        _vectors = np.stack(embeds)
        # Normalize for cosine similarity (dot product of normalized vectors = cosine sim)
        norms = np.linalg.norm(_vectors, axis=1, keepdims=True)
        # avoid div by zero
        norms[norms == 0] = 1
        _vectors = _vectors / norms
    else:
        _vectors = np.empty((0, 768))
        
    print("[RAG] Vector store ready.")

def retrieve_guideline(patient_transcript: list[dict], threshold=0.55) -> str:
    """
    Looks at what the patient has said so far, finds the most relevant clinical
    guideline from the vector store, and returns it. Returns an empty string if
    no guideline matches closely enough.
    """
    _load_and_embed()
    if not _guidelines or _vectors is None or len(_vectors) == 0:
        return ""

    # Extract only the patient's words to find out what they are complaining about
    patient_text = " ".join([t["text"] for t in patient_transcript if t["role"] == "patient"])
    if not patient_text.strip():
        return ""

    query_vec = get_embedding(patient_text)
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        return ""
    query_vec = query_vec / norm

    # Compute cosine similarity
    similarities = np.dot(_vectors, query_vec)
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score > threshold:
        matched = _guidelines[best_idx]
        print(f"[RAG] Matched '{matched['condition']}' with score {best_score:.2f}")
        return matched["protocol"]
    
    return ""
