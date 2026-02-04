#!/usr/bin/env python3
"""
hybrid_rag.py — Combine Lexical (TF-IDF) and Dense (Embedding) Retrieval

WHAT THIS IS
------------
A single-file demo showing how hybrid search combines keyword-based and
semantic retrieval for better results. Implements:
  1) TF-IDF (lexical/keyword) retrieval — good for exact terms, rare words
  2) Dense embedding retrieval — good for semantic similarity
  3) Hybrid via Reciprocal Rank Fusion (RRF) — best of both worlds

WHY THIS EXISTS
---------------
Pure embedding search can miss exact keyword matches (names, codes, acronyms).
Pure keyword search misses semantically related content. Hybrid search combines
both signals, often outperforming either alone.

REQUIREMENTS
------------
- Python 3.9+
- `pip install requests scikit-learn numpy`
- Ollama running locally: `ollama serve`
- Models: `ollama pull llama3.2` and `ollama pull nomic-embed-text`

HOW TO RUN
----------
# Compare all retrieval methods
python hybrid_rag.py "What is RAG?"

# Test specific method
python hybrid_rag.py "How do I reset my password?" --method hybrid

# Enable query expansion
python hybrid_rag.py "vacation policy" --expand-query

# Adjust fusion constant
python hybrid_rag.py "API endpoints" --rrf-k 30

# Verbose mode shows all scores
python hybrid_rag.py "chunking strategies" --verbose
"""

import argparse
import math
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import requests
from requests.exceptions import RequestException


# -----------------------------
# Terminal Colors
# -----------------------------
class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.BOLD = cls.DIM = cls.RESET = ''


if not sys.stdout.isatty():
    Colors.disable()


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 55}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 55}{Colors.RESET}")


def print_success(text: str):
    print(f"{Colors.GREEN}[OK]{Colors.RESET} {text}")


def print_error(text: str):
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {text}")


def print_progress(text: str):
    print(f"{Colors.DIM}>>>{Colors.RESET} {text}")


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_GEN_MODEL = "llama3.2"
DEFAULT_EMB_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 3
RRF_K = 60  # Reciprocal Rank Fusion constant (higher = less weight to rank)


# -----------------------------
# Sample Knowledge Base (same domain as simple-rag demos)
# -----------------------------
DOCUMENTS = {
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
Start around 300-800 characters or 150-400 tokens. Overlap (e.g., 10-20%) can help preserve context
across boundaries. For structured docs (Markdown, HTML), prefer semantic chunking by headings/sections.
""",
    "doc5_embeddings_models": """
Embedding models map text to vectors capturing semantic meaning. Domain- or instruction-tuned embedding
models sometimes improve retrieval quality. Always store the embed model + version you used to generate
your index—mixing models can degrade results. Normalize or standardize vectors (implicitly handled by cosine).
""",
    "doc6_bm25_tfidf": """
BM25 and TF-IDF are lexical retrieval methods based on term frequency. They excel at exact keyword
matching and rare term retrieval. TF-IDF weights terms by inverse document frequency. BM25 adds
saturation and document length normalization. Use for code, names, or domain jargon.
""",
    "doc7_hybrid_search": """
Hybrid search blends lexical (BM25/TF-IDF) and dense (embedding) signals. This often improves retrieval
of names, codes, or rare terms while keeping semantic relevance strong. Reciprocal Rank Fusion (RRF)
is a simple way to combine ranked lists: RRF_score = sum(1 / (k + rank)) across methods.
""",
    "doc8_password_reset": """
To reset your password in TechFlow: 1) Go to login.techflow.com, 2) Click 'Forgot Password',
3) Enter your email address, 4) Check your email for reset link (may take up to 10 minutes),
5) Create new password with at least 8 characters including numbers and symbols.
""",
    "doc9_vacation_policy": """
Full-time employees receive 15 days PTO annually, increasing to 20 days after 3 years and
25 days after 5 years. Vacation requests must be submitted at least 2 weeks in advance.
Maximum 5 consecutive days without manager approval. Unused PTO does not roll over.
""",
    "doc10_query_expansion": """
Query expansion improves retrieval by adding synonyms or related terms to the original query.
An LLM can generate expanded queries: "What are the vacation rules?" might expand to
"vacation policy PTO time off leave days annual". This helps bridge vocabulary mismatches.
""",
}


# -----------------------------
# Utilities
# -----------------------------
def assert_ollama_up(host: str) -> None:
    """Verify Ollama is reachable."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
    except RequestException as e:
        raise SystemExit(f"[!] Could not reach Ollama at {host}. Is `ollama serve` running?\n{e}")


def normalize_ws(text: str) -> str:
    """Collapse whitespace and trim."""
    return " ".join(text.split()).strip()


def embed_text(host: str, model: str, text: str) -> List[float]:
    """Get embedding vector from Ollama."""
    url = f"{host}/api/embeddings"

    # Try 'prompt' key first, fall back to 'input'
    for key in ["prompt", "input"]:
        try:
            resp = requests.post(url, json={"model": model, key: text}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "embedding" in data:
                return data["embedding"]
        except RequestException:
            continue

    raise RuntimeError(f"Failed to get embedding from {host}")


def cosine_sim(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)


def chat_ollama(host: str, model: str, prompt: str) -> str:
    """Simple chat completion with Ollama."""
    url = f"{host}/api/generate"
    resp = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# -----------------------------
# Retrieval Methods
# -----------------------------
class HybridRAG:
    """
    Hybrid retrieval combining TF-IDF and dense embeddings.

    Implements three retrieval strategies:
    - keyword: TF-IDF based lexical matching
    - embedding: Dense vector similarity
    - hybrid: Reciprocal Rank Fusion of both
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        gen_model: str = DEFAULT_GEN_MODEL,
        emb_model: str = DEFAULT_EMB_MODEL,
        rrf_k: int = RRF_K
    ):
        self.host = host
        self.gen_model = gen_model
        self.emb_model = emb_model
        self.rrf_k = rrf_k

        # Storage
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []

        # TF-IDF components
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.tfidf_matrix = None

        # Embedding components
        self.embeddings: List[List[float]] = []

    def add_documents(self, documents: Dict[str, str]) -> None:
        """Index documents for both retrieval methods."""
        print_progress(f"Indexing {Colors.BOLD}{len(documents)}{Colors.RESET} documents...")

        self.doc_ids = list(documents.keys())
        self.doc_texts = [normalize_ws(text) for text in documents.values()]

        # Build TF-IDF index
        print(f"  {Colors.DIM}Building TF-IDF index...{Colors.RESET}")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.doc_texts)

        # Build embedding index
        print(f"  {Colors.DIM}Computing embeddings...{Colors.RESET}")
        self.embeddings = [
            embed_text(self.host, self.emb_model, text)
            for text in self.doc_texts
        ]

        print_success(f"Indexed {Colors.BOLD}{len(self.doc_ids)}{Colors.RESET} documents")
        print(f"  {Colors.DIM}TF-IDF vocabulary: {len(self.tfidf_vectorizer.vocabulary_)} terms{Colors.RESET}")

    def retrieve_keyword(self, query: str, top_k: int) -> List[Tuple[str, str, float]]:
        """TF-IDF based keyword retrieval."""
        query_vec = self.tfidf_vectorizer.transform([query.lower()])
        similarities = sklearn_cosine(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((
                    self.doc_ids[idx],
                    self.doc_texts[idx],
                    float(similarities[idx])
                ))

        return results

    def retrieve_embedding(self, query: str, top_k: int) -> List[Tuple[str, str, float]]:
        """Dense embedding retrieval."""
        query_vec = embed_text(self.host, self.emb_model, query)

        # Compute similarities
        similarities = [cosine_sim(query_vec, emb) for emb in self.embeddings]

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((
                self.doc_ids[idx],
                self.doc_texts[idx],
                similarities[idx]
            ))

        return results

    def retrieve_hybrid(self, query: str, top_k: int) -> List[Tuple[str, str, float]]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) for each method
        where k is a constant (default 60) that controls rank importance.
        """
        # Get results from both methods (more than top_k to have overlap)
        n_retrieve = min(top_k * 2, len(self.doc_ids))
        keyword_results = self.retrieve_keyword(query, n_retrieve)
        embedding_results = self.retrieve_embedding(query, n_retrieve)

        # Build rank maps (doc_id -> rank, 1-indexed)
        keyword_ranks = {doc_id: rank + 1 for rank, (doc_id, _, _) in enumerate(keyword_results)}
        embedding_ranks = {doc_id: rank + 1 for rank, (doc_id, _, _) in enumerate(embedding_results)}

        # Compute RRF scores for all docs that appear in either list
        all_doc_ids = set(keyword_ranks.keys()) | set(embedding_ranks.keys())
        rrf_scores = {}

        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in keyword_ranks:
                score += 1.0 / (self.rrf_k + keyword_ranks[doc_id])
            if doc_id in embedding_ranks:
                score += 1.0 / (self.rrf_k + embedding_ranks[doc_id])
            rrf_scores[doc_id] = score

        # Sort by RRF score and return top-k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            idx = self.doc_ids.index(doc_id)
            results.append((doc_id, self.doc_texts[idx], score))

        return results

    def expand_query(self, query: str) -> str:
        """Use LLM to expand query with related terms."""
        prompt = f"""Given this search query, generate an expanded version that includes synonyms and related terms to improve search retrieval. Keep it concise (under 50 words).

Original query: {query}

Expanded query:"""

        expanded = chat_ollama(self.host, self.gen_model, prompt)
        # Combine original and expanded
        return f"{query} {expanded}"

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = DEFAULT_TOP_K,
        expand: bool = False
    ) -> List[Tuple[str, str, float]]:
        """Main retrieval entry point."""
        if expand:
            query = self.expand_query(query)
            print(f"  Expanded query: {query[:100]}...")

        if method == "keyword":
            return self.retrieve_keyword(query, top_k)
        elif method == "embedding":
            return self.retrieve_embedding(query, top_k)
        elif method == "hybrid":
            return self.retrieve_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")

    def generate_answer(self, query: str, context_docs: List[Tuple[str, str, float]]) -> str:
        """Generate answer using retrieved context."""
        if not context_docs:
            return "No relevant documents found."

        # Build context string
        context_parts = []
        for i, (doc_id, text, score) in enumerate(context_docs, 1):
            context_parts.append(f"[{i}] {text[:500]}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer isn't in the context, say so. Use [1], [2] citations when relevant.

Context:
{context}

Question: {query}

Answer:"""

        return chat_ollama(self.host, self.gen_model, prompt)


# -----------------------------
# Comparison & Evaluation
# -----------------------------
def compare_methods(
    rag: HybridRAG,
    query: str,
    top_k: int,
    expand: bool,
    verbose: bool
) -> Dict:
    """Compare all retrieval methods on a single query."""
    methods = ["keyword", "embedding", "hybrid"]
    results = {}

    for method in methods:
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}Method: {Colors.YELLOW}{method.upper()}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 50}{Colors.RESET}")

        t0 = time.time()
        docs = rag.retrieve(query, method=method, top_k=top_k, expand=expand)
        elapsed = time.time() - t0

        print_success(f"Retrieved {Colors.BOLD}{len(docs)}{Colors.RESET} documents in {elapsed:.2f}s")
        for i, (doc_id, text, score) in enumerate(docs, 1):
            preview = text[:80].replace('\n', ' ')
            print(f"  {Colors.CYAN}{i}.{Colors.RESET} [{Colors.BOLD}{doc_id}{Colors.RESET}] score={Colors.GREEN}{score:.4f}{Colors.RESET}")
            if verbose:
                print(f"     {Colors.DIM}{preview}...{Colors.RESET}")

        results[method] = {
            "docs": [(doc_id, score) for doc_id, _, score in docs],
            "elapsed": elapsed
        }

    return results


def print_comparison_summary(results: Dict) -> None:
    """Print summary showing overlap and differences."""
    print_header("Comparison Summary")

    # Gather doc sets
    keyword_docs = set(doc_id for doc_id, _ in results["keyword"]["docs"])
    embedding_docs = set(doc_id for doc_id, _ in results["embedding"]["docs"])
    hybrid_docs = set(doc_id for doc_id, _ in results["hybrid"]["docs"])

    print(f"  {Colors.BOLD}Keyword retrieved:{Colors.RESET}   {keyword_docs}")
    print(f"  {Colors.BOLD}Embedding retrieved:{Colors.RESET} {embedding_docs}")
    print(f"  {Colors.BOLD}Hybrid retrieved:{Colors.RESET}    {hybrid_docs}")

    # Overlap analysis
    all_overlap = keyword_docs & embedding_docs & hybrid_docs
    keyword_only = keyword_docs - embedding_docs - hybrid_docs
    embedding_only = embedding_docs - keyword_docs - hybrid_docs

    print(f"\n  {Colors.GREEN}In all three:{Colors.RESET}        {all_overlap or '{none}'}")
    print(f"  {Colors.YELLOW}Keyword only:{Colors.RESET}        {keyword_only or '{none}'}")
    print(f"  {Colors.BLUE}Embedding only:{Colors.RESET}      {embedding_only or '{none}'}")


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid RAG: Compare keyword, embedding, and hybrid retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("query", type=str, help="Search query")
    p.add_argument("--method", choices=["keyword", "embedding", "hybrid", "all"], default="all",
                   help="Retrieval method to use")
    p.add_argument("--host", default=DEFAULT_HOST, help="Ollama host URL")
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL, help="Generation model")
    p.add_argument("--embed-model", default=DEFAULT_EMB_MODEL, help="Embedding model")
    p.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Number of documents to retrieve")
    p.add_argument("--rrf-k", type=int, default=RRF_K, help="RRF constant (higher = less rank weight)")
    p.add_argument("--expand-query", action="store_true", help="Expand query using LLM")
    p.add_argument("--generate", action="store_true", help="Generate answer after retrieval")
    p.add_argument("--verbose", action="store_true", help="Show document previews")
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    print_header("Hybrid RAG Demo")
    print(f"  {Colors.BOLD}Query:{Colors.RESET} {args.query}")
    print(f"  {Colors.BOLD}Model:{Colors.RESET} {args.gen_model} {Colors.DIM}(gen){Colors.RESET}, {args.embed_model} {Colors.DIM}(emb){Colors.RESET}")

    # Check Ollama
    assert_ollama_up(args.host)

    # Initialize and index
    rag = HybridRAG(
        host=args.host,
        gen_model=args.gen_model,
        emb_model=args.embed_model,
        rrf_k=args.rrf_k
    )
    rag.add_documents(DOCUMENTS)

    # Run retrieval
    if args.method == "all":
        results = compare_methods(rag, args.query, args.k, args.expand_query, args.verbose)
        print_comparison_summary(results)

        # Generate answer using hybrid results if requested
        if args.generate:
            print_header("Generated Answer (hybrid retrieval)")
            docs = rag.retrieve(args.query, method="hybrid", top_k=args.k)
            answer = rag.generate_answer(args.query, docs)
            print(f"\n{answer}")
    else:
        # Single method
        docs = rag.retrieve(args.query, method=args.method, top_k=args.k, expand=args.expand_query)

        print_success(f"Retrieved {Colors.BOLD}{len(docs)}{Colors.RESET} documents")
        for i, (doc_id, text, score) in enumerate(docs, 1):
            print(f"\n{Colors.CYAN}[{i}]{Colors.RESET} {Colors.BOLD}{doc_id}{Colors.RESET} {Colors.DIM}(score: {score:.4f}){Colors.RESET}")
            if args.verbose:
                print(f"    {Colors.DIM}{text[:200]}...{Colors.RESET}")

        if args.generate:
            print_header("Generated Answer")
            answer = rag.generate_answer(args.query, docs)
            print(f"\n{answer}")


if __name__ == "__main__":
    main(sys.argv[1:])
