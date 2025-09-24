#!/usr/bin/env python3
"""
mini_rag_ollama.py — a tiny, well-documented Retrieval-Augmented Generation (RAG) demo for Ollama

WHAT THIS IS
------------
A single-file Python script that:
  1) builds a trivial in-memory "vector store" from a handful of example documents,
  2) embeds those document chunks via Ollama's /api/embeddings,
  3) retrieves the top-k chunks for a given question using cosine similarity, and
  4) asks an Ollama chat model to answer ONLY using the retrieved context.

WHY THIS EXISTS
---------------
It's meant to be the smallest possible reference you can read end-to-end to understand the moving parts.
No external DBs, no frameworks—just plain requests, a little math, and useful comments.

REQUIREMENTS
------------
- Python 3.9+
- `pip install requests`
- Ollama running locally:
    ollama serve
- Models (pick small ones for speed):
    ollama pull llama3.1
    ollama pull nomic-embed-text

HOW TO RUN
----------
# Ask a question with defaults
python mini_rag_ollama.py "How do I use embeddings for RAG?"

# Change top-k or chunk size
python mini_rag_ollama.py "What is cosine similarity?" --k 5 --chunk 350

# Choose models / host
python mini_rag_ollama.py "Tips for chunking?" --gen-model llama3.1 --embed-model nomic-embed-text --host http://localhost:11434

DESIGN NOTES
------------
- We deliberately embed texts one-by-one for clarity/compatibility. Ollama's embeddings endpoint commonly accepts a single prompt per call.
- We use cosine similarity for retrieval (the most common choice).
- We keep everything in RAM. For real apps, swap in FAISS/Chroma/pgvector.
- The prompt asks the model to answer *only* from provided context and to admit "I don't know" when needed.

NEXT STEPS
----------
- Replace DOCUMENTS with your own content (or load files) and rerun.
- Add a real vector DB, a tokenizer-aware chunker, and MMR or hybrid (keyword + dense) retrieval.
- Add caching/persistence for embeddings so you don’t recompute every run.

FAQ
---
Q: My Ollama build expects "input" instead of "prompt" for /api/embeddings.
A: Flip EMBED_PAYLOAD_KEY below or use the built-in fallback try/except in `embed()`.

Q: What about streaming?
A: For simplicity, we request a non-streaming chat response. You can adapt to stream partial deltas if you prefer.
"""

import argparse
import math
import sys
import time
from typing import Dict, List, Tuple, Iterable

import requests
from requests.exceptions import RequestException

# -----------------------------
# Configuration (CLI overrides most of this)
# -----------------------------
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_GEN_MODEL = "llama3.2"           # swap for any local chat model (qwen2, phi3, etc.)
DEFAULT_EMB_MODEL = "nomic-embed-text"   # swap for any local embedding model
DEFAULT_TOP_K = 3
DEFAULT_CHUNK_CHARS = 400

# Some Ollama builds prefer "prompt", others "input".
# We try "prompt" first, then fall back to "input" if needed.
EMBED_PAYLOAD_KEY = "prompt"


# -----------------------------
# A larger, documented toy corpus
# (Each entry is <~400-900 chars so chunking also demonstrates)
# -----------------------------
DOCUMENTS: Dict[str, str] = {
    "doc1_ollama_endpoints": """
Ollama exposes simple HTTP endpoints on localhost. Common routes include /api/chat
for multi-turn chat, /api/generate for single-turn completion, and /api/embeddings
for vector generation. Start the server with `ollama serve`, and pull models with
`ollama pull <model>`. When experimenting, prefer smaller, quantized models for speed.
""",
    "doc2_what_is_rag": """
Retrieval-Augmented Generation (RAG) is a pattern that injects external knowledge into
a language model at inference time. Steps: (1) chunk your documents, (2) embed them
into vectors, (3) embed the user question, (4) retrieve the top-k similar chunks, and
(5) prompt a model with those chunks as context. This keeps answers grounded and reduces hallucinations.
""",
    "doc3_cosine_similarity": """
Cosine similarity measures the angle between two vectors and is computed as dot(a,b)/(||a||*||b||).
Values range from -1 to 1; higher means more similar. For normalized embedding vectors, cosine is a
good default similarity measure in semantic search and RAG retrieval.
""",
    "doc4_chunking_strategies": """
Chunking strategy matters. Overly large chunks dilute relevance; overly small chunks lose context.
Start around 300–800 characters or 150–400 tokens. Overlap (e.g., 10–20%) can help preserve context
across boundaries. For structured docs (Markdown, HTML), prefer semantic chunking by headings/sections.
""",
    "doc5_embeddings_models": """
Embedding models map text to vectors capturing semantic meaning. Domain- or instruction-tuned embedding
models sometimes improve retrieval quality. Always store the embed model + version you used to generate
your index—mixing models can degrade results. Normalize or standardize vectors (implicitly handled by cosine).
""",
    "doc6_prompt_template": """
Your RAG prompt should instruct the model to answer strictly from provided context. Ask it to cite which
snippets it used (e.g., [1], [2]) and to say “I don’t know” if the answer isn’t present. This reduces
hallucinations and makes evaluation easier.
""",
    "doc7_streaming_vs_nonstreaming": """
Non-streaming responses are convenient for simple demos. Streaming helps with perceived latency and UX,
especially for longer completions. Ollama supports streaming for /api/generate and /api/chat with `stream: true`.
""",
    "doc8_retrieval_pitfalls": """
Common retrieval pitfalls: (1) index built with different embed model than query time, (2) bad chunk sizes,
(3) missing pre/post-processing (lowercasing, trimming repeated whitespace), (4) ignoring exact term matches
when needed—hybrid search (keyword + vector) can help.
""",
    "doc9_eval": """
Evaluate RAG with grounded metrics (faithfulness, answer relevance) and retrieval metrics (Recall@k, MRR).
Human spot checks are invaluable—verify the model actually used retrieved evidence rather than prior knowledge.
""",
    "doc10_citations": """
Including inline citations like [1], [2] that map to retrieved chunks helps users trust answers and debug failures.
Show document IDs and similarity scores for transparency during development.
""",
    "doc11_env_setup": """
Local dev loop: (1) run `ollama serve`, (2) pull a small chat model (e.g., llama3.1) and an embedding model
(e.g., nomic-embed-text), (3) test the /api endpoints with curl or a small script to confirm connectivity,
(4) iterate on chunking and prompts.
""",
    "doc12_vector_stores": """
RAM-based lists are fine for toy demos. For larger corpora, use a vector database or library like FAISS, Chroma,
or pgvector. These support fast ANN search, persistence, and metadata filters (source, timestamp, tags).
""",
    "doc13_guardrails": """
Guardrails for production: timeouts and retries on network calls, input size checks, prompt length budgeting,
PII scrubs, and clear fallbacks when retrieval returns nothing.
""",
    "doc14_hybrid_search": """
Hybrid search blends lexical (BM25/keyword) and dense (embeddings) signals. This often improves retrieval of
names, codes, or rare terms while keeping semantic relevance strong.
""",
}

# -----------------------------
# Utilities
# -----------------------------
def assert_ollama_up(host: str) -> None:
    """
    Quick health check: hit /api/tags to verify the daemon is reachable.
    Raises an exception if not reachable.
    """
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
    except RequestException as e:
        raise SystemExit(
            f"[!] Could not reach Ollama at {host}. Is `ollama serve` running?\n{e}"
        )

def normalize_ws(text: str) -> str:
    """Collapse internal whitespace and trim ends."""
    return " ".join(text.split()).strip()

def chunk_text(text: str, max_chars: int) -> List[str]:
    """
    Trivial character-based chunker. For real apps, switch to a tokenizer-aware
    or semantic chunker (e.g., split by headings, sentences, or paragraphs).
    """
    text = normalize_ws(text)
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def embed_text(host: str, model: str, payload_key: str, text: str) -> List[float]:
    """
    Calls Ollama's /api/embeddings for a single string.
    Tries `payload_key` first (usually 'prompt'), falls back to 'input' on error.
    """
    url = f"{host}/api/embeddings"

    # 1) First attempt with configured key (default: 'prompt')
    try:
        resp = requests.post(url, json={"model": model, payload_key: text, "options": {}}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "embedding" in data:
            return data["embedding"]
    except RequestException:
        pass  # we try the fallback next

    # 2) Fallback attempt with the alternate key
    alt_key = "input" if payload_key == "prompt" else "prompt"
    resp = requests.post(url, json={"model": model, alt_key: text, "options": {}}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]

def embed_many(host: str, model: str, payload_key: str, texts: Iterable[str]) -> List[List[float]]:
    """
    Embed a list of texts. Kept simple/serial for clarity and compatibility.
    """
    return [embed_text(host, model, payload_key, t) for t in texts]

def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)

def top_k(
    query_vec: List[float],
    db: List[Tuple[str, str, List[float]]],
    k: int,
) -> List[Tuple[float, str, str]]:
    """
    Returns top-k (similarity, doc_id, chunk_text) triples.
    """
    scored = [(cosine(query_vec, vec), doc_id, chunk) for (doc_id, chunk, vec) in db]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]

def chat_nonstream(host: str, model: str, messages: List[Dict[str, str]]) -> str:
    """
    Calls Ollama /api/chat with stream=False and returns the assistant content.
    """
    url = f"{host}/api/chat"
    resp = requests.post(
        url,
        json={"model": model, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]

# -----------------------------
# Building the (toy) vector store
# -----------------------------
def build_store(
    host: str,
    emb_model: str,
    max_chars: int,
    payload_key: str = EMBED_PAYLOAD_KEY,
) -> List[Tuple[str, str, List[float]]]:
    """
    Returns a list of (doc_id, chunk_text, embedding_vector).
    """
    chunks: List[Tuple[str, str]] = []
    for doc_id, text in DOCUMENTS.items():
        for ch in chunk_text(text, max_chars=max_chars):
            chunks.append((doc_id, ch))

    vecs = embed_many(host, emb_model, payload_key, (ch for _, ch in chunks))
    db = [(doc_id, ch, vec) for (doc_id, ch), vec in zip(chunks, vecs)]
    return db

# -----------------------------
# RAG pipeline
# -----------------------------
def rag_answer(
    host: str,
    gen_model: str,
    emb_model: str,
    question: str,
    db: List[Tuple[str, str, List[float]]],
    k: int,
    verbose: bool = False,
) -> str:
    """
    1) Embed the question
    2) Retrieve top-k similar chunks
    3) Ask the chat model to answer ONLY from those chunks
    """
    q_vec = embed_many(host, emb_model, EMBED_PAYLOAD_KEY, [question])[0]
    hits = top_k(q_vec, db, k=k)

    # Build a context block with numbered snippets like [1], [2], ...
    context_blocks: List[str] = []
    citations: List[str] = []
    for i, (score, doc_id, chunk) in enumerate(hits, start=1):
        context_blocks.append(f"[{i}] {chunk.strip()}")
        citations.append(f"[{i}:{doc_id}, score={score:.3f}]")

    context = "\n\n".join(context_blocks)

    # System + user instruction. Force grounding.
    user_prompt = (
        "You are a helpful assistant. Answer the user using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    messages = [
        {"role": "system", "content": "Be concise. Use [1], [2] citations when relevant."},
        {"role": "user", "content": user_prompt},
    ]

    answer = chat_nonstream(host, gen_model, messages).strip()

    if verbose:
        debug = "\n".join(f"  {c}" for c in citations)
        answer += f"\n\n[debug] Sources:\n{debug}"
    else:
        answer += "\n\nSources: " + "  ".join(citations)

    return answer

# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A tiny, well-documented RAG demo that talks to a local Ollama server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("question", type=str, help="Your question in quotes.")
    p.add_argument("--host", type=str, default=DEFAULT_HOST, help="Ollama host URL.")
    p.add_argument("--gen-model", type=str, default=DEFAULT_GEN_MODEL, help="Chat model.")
    p.add_argument("--embed-model", type=str, default=DEFAULT_EMB_MODEL, help="Embedding model.")
    p.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Top-k chunks to retrieve.")
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_CHARS, help="Max characters per chunk.")
    p.add_argument("--verbose", action="store_true", help="Include similarity scores and doc IDs.")
    return p.parse_args(argv)

def main(argv: List[str]) -> None:
    args = parse_args(argv)

    # 1) Connectivity check (fast fail if daemon is down)
    assert_ollama_up(args.host)

    # 2) Build in-memory index from the example documents
    print(f"Building in-memory vector store (chunks ~{args.chunk} chars)...")
    t0 = time.time()
    db = build_store(args.host, args.embed_model, args.chunk)
    print(f"Built {len(db)} chunks in {time.time() - t0:.2f}s")

    # 3) Retrieve + generate
    print("Retrieving + generating...")
    t1 = time.time()
    out = rag_answer(
        host=args.host,
        gen_model=args.gen_model,
        emb_model=args.embed_model,
        question=args.question,
        db=db,
        k=args.k,
        verbose=args.verbose,
    )
    dt = time.time() - t1

    print("\n" + "=" * 80)
    print(out)
    print("=" * 80)
    print(f"Took {dt:.2f}s")

if __name__ == "__main__":
    main(sys.argv[1:])
