#!/usr/bin/env python3
"""
semantic_chunker.py — Compare Document Chunking Strategies for RAG

WHAT THIS IS
------------
A single-file demo comparing different text chunking approaches:
  1) Character-based: Fixed character count (naive)
  2) Sentence-aware: Break at sentence boundaries
  3) Paragraph-aware: Break at paragraph boundaries
  4) Section-aware: Break at markdown headers/sections

Each strategy can use sliding window overlap to preserve context.

WHY THIS EXISTS
---------------
Chunking quality directly affects RAG retrieval. Bad chunks = irrelevant results.
This demo helps you understand the trade-offs and see the differences visually.

REQUIREMENTS
------------
- Python 3.9+
- `pip install requests`
- Ollama running locally: `ollama serve`
- Models: `ollama pull llama3.2` and `ollama pull nomic-embed-text`

HOW TO RUN
----------
# Compare all chunking strategies on sample document
python semantic_chunker.py

# Test specific strategy
python semantic_chunker.py --strategy sentence

# Adjust chunk size and overlap
python semantic_chunker.py --max-chars 500 --overlap 100

# Run retrieval comparison with a query
python semantic_chunker.py --query "How do I install the software?"

# Show full chunk contents
python semantic_chunker.py --verbose
"""

import argparse
import math
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
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


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_EMB_MODEL = "nomic-embed-text"
DEFAULT_MAX_CHARS = 400
DEFAULT_OVERLAP = 50


# -----------------------------
# Sample Document (Markdown with sections)
# -----------------------------
SAMPLE_DOCUMENT = """# TechFlow Installation Guide

Welcome to TechFlow! This guide will help you get started with our software.

## System Requirements

Before installing TechFlow, ensure your system meets these requirements:

- Operating System: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- RAM: Minimum 8GB, recommended 16GB
- Disk Space: At least 2GB free
- Internet connection required for activation

TechFlow runs best on systems with SSD storage and dedicated graphics cards for visualization features.

## Installation Steps

Follow these steps to install TechFlow on your computer:

1. Download the installer from https://techflow.com/download
2. Run the installer and accept the license agreement
3. Choose your installation directory (default is recommended)
4. Wait for the installation to complete (usually 5-10 minutes)
5. Launch TechFlow from your applications menu

If you encounter any errors during installation, check the troubleshooting section below.

## Configuration

After installation, configure TechFlow for optimal performance:

### Database Setup

TechFlow requires a database connection. You can use:
- SQLite (default, no setup required)
- PostgreSQL (recommended for teams)
- MySQL (legacy support)

To configure PostgreSQL, edit the config.yaml file and set the database URL.

### API Keys

Some features require API keys:
- OpenAI API key for AI features
- Stripe API key for billing integration
- SendGrid API key for email notifications

Store API keys in environment variables, never in code.

## Troubleshooting

Common issues and solutions:

### Installation Fails

If installation fails, try these steps:
1. Run the installer as administrator
2. Disable antivirus temporarily
3. Check disk space availability
4. Download a fresh copy of the installer

### Connection Errors

If you see connection errors:
- Check your internet connection
- Verify firewall settings allow TechFlow
- Try using a VPN if your network blocks certain ports

### Performance Issues

For slow performance:
- Close other resource-intensive applications
- Increase allocated RAM in settings
- Clear the application cache
- Update to the latest version

## Getting Help

If you need additional assistance:
- Visit our documentation at docs.techflow.com
- Join our community forum at community.techflow.com
- Contact support at support@techflow.com
- Check our YouTube channel for video tutorials
"""


# -----------------------------
# Chunking Strategies
# -----------------------------
@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    start_char: int
    end_char: int
    strategy: str
    index: int


def chunk_by_characters(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP
) -> List[Chunk]:
    """
    Naive character-based chunking with sliding window.

    Pros: Simple, predictable chunk sizes
    Cons: Can split mid-word, mid-sentence, loses context
    """
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                start_char=start,
                end_char=end,
                strategy="character",
                index=index
            ))
            index += 1

        # Move forward with overlap
        start = end - overlap if end < len(text) else len(text)

    return chunks


def chunk_by_sentences(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP
) -> List[Chunk]:
    """
    Sentence-aware chunking: break at sentence boundaries.

    Pros: Preserves complete thoughts, better semantic units
    Cons: Variable chunk sizes, some sentences may be very long
    """
    # Split into sentences (handling common abbreviations)
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)

    chunks = []
    current_chunk = []
    current_length = 0
    start_char = 0
    index = 0
    char_pos = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_len = len(sentence)

        # If single sentence exceeds max, add it as its own chunk
        if sentence_len > max_chars and current_chunk:
            # Flush current chunk first
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_char=start_char,
                end_char=char_pos,
                strategy="sentence",
                index=index
            ))
            index += 1
            current_chunk = []
            current_length = 0
            start_char = char_pos

        # If adding this sentence would exceed max, start new chunk
        if current_length + sentence_len + 1 > max_chars and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_char=start_char,
                end_char=char_pos,
                strategy="sentence",
                index=index
            ))
            index += 1

            # Keep last sentence(s) for overlap if they fit
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break

            current_chunk = overlap_sentences
            current_length = overlap_len
            start_char = char_pos - overlap_len

        current_chunk.append(sentence)
        current_length += sentence_len + 1
        char_pos += sentence_len + 1

    # Add final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(Chunk(
            text=chunk_text,
            start_char=start_char,
            end_char=len(text),
            strategy="sentence",
            index=index
        ))

    return chunks


def chunk_by_paragraphs(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP
) -> List[Chunk]:
    """
    Paragraph-aware chunking: break at double newlines.

    Pros: Preserves topical units, natural document structure
    Cons: Paragraphs can be very long or very short
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = []
    current_length = 0
    start_char = 0
    index = 0
    char_pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)

        # If adding this paragraph would exceed max, start new chunk
        if current_length + para_len + 2 > max_chars and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_char=start_char,
                end_char=char_pos,
                strategy="paragraph",
                index=index
            ))
            index += 1

            # Keep last paragraph for overlap if it fits
            if current_chunk and len(current_chunk[-1]) <= overlap:
                current_chunk = [current_chunk[-1]]
                current_length = len(current_chunk[0]) + 2
                start_char = char_pos - current_length
            else:
                current_chunk = []
                current_length = 0
                start_char = char_pos

        current_chunk.append(para)
        current_length += para_len + 2
        char_pos += para_len + 2

    # Add final chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append(Chunk(
            text=chunk_text,
            start_char=start_char,
            end_char=len(text),
            strategy="paragraph",
            index=index
        ))

    return chunks


def chunk_by_sections(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP
) -> List[Chunk]:
    """
    Section-aware chunking: break at markdown headers.

    Pros: Preserves document structure, headers provide context
    Cons: Sections can vary widely in size
    """
    # Split on markdown headers (keeping the header with content)
    section_pattern = r'(^#{1,3}\s+.+$)'
    parts = re.split(section_pattern, text, flags=re.MULTILINE)

    # Reconstruct sections (header + content pairs)
    sections = []
    current_header = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r'^#{1,3}\s+', part):
            current_header = part
        else:
            section_text = f"{current_header}\n\n{part}" if current_header else part
            sections.append(section_text)
            current_header = ""

    chunks = []
    index = 0
    char_pos = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # If section fits in one chunk, use it as-is
        if len(section) <= max_chars:
            chunks.append(Chunk(
                text=section,
                start_char=char_pos,
                end_char=char_pos + len(section),
                strategy="section",
                index=index
            ))
            index += 1
        else:
            # Section too long, sub-chunk by paragraphs
            sub_chunks = chunk_by_paragraphs(section, max_chars, overlap)
            for sub in sub_chunks:
                chunks.append(Chunk(
                    text=sub.text,
                    start_char=char_pos + sub.start_char,
                    end_char=char_pos + sub.end_char,
                    strategy="section",
                    index=index
                ))
                index += 1

        char_pos += len(section) + 2

    return chunks


STRATEGIES = {
    "character": chunk_by_characters,
    "sentence": chunk_by_sentences,
    "paragraph": chunk_by_paragraphs,
    "section": chunk_by_sections,
}


# -----------------------------
# Embedding & Retrieval
# -----------------------------
def assert_ollama_up(host: str) -> None:
    """Verify Ollama is reachable."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
    except RequestException as e:
        raise SystemExit(f"[!] Could not reach Ollama at {host}. Is `ollama serve` running?\n{e}")


def embed_text(host: str, model: str, text: str) -> List[float]:
    """Get embedding vector from Ollama."""
    url = f"{host}/api/embeddings"
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
    """Compute cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)


def retrieve_top_k(
    query: str,
    chunks: List[Chunk],
    host: str,
    emb_model: str,
    top_k: int = 3
) -> List[Tuple[Chunk, float]]:
    """Retrieve top-k chunks by embedding similarity."""
    query_vec = embed_text(host, emb_model, query)

    # Compute similarities
    scored = []
    for chunk in chunks:
        chunk_vec = embed_text(host, emb_model, chunk.text)
        sim = cosine_sim(query_vec, chunk_vec)
        scored.append((chunk, sim))

    # Sort and return top-k
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# -----------------------------
# Visualization & Comparison
# -----------------------------
def print_chunks(chunks: List[Chunk], verbose: bool = False) -> None:
    """Print chunk information."""
    for chunk in chunks:
        preview = chunk.text[:60].replace('\n', ' ')
        print(f"  {Colors.CYAN}[{chunk.index}]{Colors.RESET} chars {chunk.start_char}-{chunk.end_char} {Colors.DIM}({len(chunk.text)} chars){Colors.RESET}")
        if verbose:
            print(f"      {Colors.DIM}\"{preview}...\"{Colors.RESET}")


def compare_strategies(
    text: str,
    max_chars: int,
    overlap: int,
    verbose: bool
) -> Dict[str, List[Chunk]]:
    """Compare all chunking strategies."""
    results = {}

    for name, strategy_fn in STRATEGIES.items():
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}Strategy: {Colors.YELLOW}{name.upper()}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 50}{Colors.RESET}")

        chunks = strategy_fn(text, max_chars, overlap)
        results[name] = chunks

        print_success(f"Created {Colors.BOLD}{len(chunks)}{Colors.RESET} chunks")

        # Statistics
        sizes = [len(c.text) for c in chunks]
        print(f"  {Colors.DIM}Min: {min(sizes)} | Max: {max(sizes)} | Avg: {sum(sizes) / len(sizes):.0f} chars{Colors.RESET}")

        print(f"\n{Colors.BOLD}Chunks:{Colors.RESET}")
        print_chunks(chunks, verbose)

    return results


def compare_retrieval(
    query: str,
    all_chunks: Dict[str, List[Chunk]],
    host: str,
    emb_model: str,
    top_k: int = 3,
    verbose: bool = False
) -> None:
    """Compare retrieval results across chunking strategies."""
    print_header("Retrieval Comparison")
    print(f"  {Colors.BOLD}Query:{Colors.RESET} \"{query}\"")

    for strategy_name, chunks in all_chunks.items():
        print(f"\n{Colors.BOLD}{Colors.YELLOW}--- {strategy_name.upper()} ---{Colors.RESET}")

        results = retrieve_top_k(query, chunks, host, emb_model, top_k)

        for i, (chunk, score) in enumerate(results, 1):
            preview = chunk.text[:80].replace('\n', ' ')
            print(f"  {Colors.CYAN}{i}.{Colors.RESET} [chunk {chunk.index}] score={Colors.GREEN}{score:.4f}{Colors.RESET}")
            if verbose:
                print(f"     {Colors.DIM}\"{preview}...\"{Colors.RESET}")


def print_summary(all_chunks: Dict[str, List[Chunk]]) -> None:
    """Print summary comparison table."""
    print_header("Summary")

    print(f"  {Colors.BOLD}{'Strategy':<12} {'Chunks':<8} {'Min':<8} {'Max':<8} {'Avg':<8}{Colors.RESET}")
    print(f"  {Colors.DIM}{'-'*44}{Colors.RESET}")

    for name, chunks in all_chunks.items():
        sizes = [len(c.text) for c in chunks]
        print(f"  {name:<12} {Colors.CYAN}{len(chunks):<8}{Colors.RESET} {min(sizes):<8} {max(sizes):<8} {sum(sizes)//len(sizes):<8}")


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare document chunking strategies for RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--strategy", choices=list(STRATEGIES.keys()) + ["all"], default="all",
                   help="Chunking strategy to use")
    p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Maximum characters per chunk")
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Overlap between chunks")
    p.add_argument("--query", type=str, help="Query to test retrieval (optional)")
    p.add_argument("--host", default=DEFAULT_HOST, help="Ollama host URL")
    p.add_argument("--embed-model", default=DEFAULT_EMB_MODEL, help="Embedding model")
    p.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    p.add_argument("--verbose", action="store_true", help="Show full chunk contents")
    p.add_argument("--input-file", type=str, help="Input file to chunk (uses sample doc if not provided)")
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    print_header("Semantic Chunking Demo")
    print(f"  {Colors.BOLD}Max chars:{Colors.RESET} {args.max_chars}")
    print(f"  {Colors.BOLD}Overlap:{Colors.RESET} {args.overlap}")

    # Load document
    if args.input_file:
        with open(args.input_file, 'r') as f:
            text = f.read()
        print_success(f"Loaded: {Colors.BOLD}{args.input_file}{Colors.RESET} ({len(text)} chars)")
    else:
        text = SAMPLE_DOCUMENT
        print(f"  {Colors.DIM}Using sample document ({len(text)} chars){Colors.RESET}")

    # Run chunking
    if args.strategy == "all":
        all_chunks = compare_strategies(text, args.max_chars, args.overlap, args.verbose)
        print_summary(all_chunks)

        # Optionally run retrieval comparison
        if args.query:
            assert_ollama_up(args.host)
            compare_retrieval(args.query, all_chunks, args.host, args.embed_model, args.top_k, args.verbose)
    else:
        # Single strategy
        strategy_fn = STRATEGIES[args.strategy]
        chunks = strategy_fn(text, args.max_chars, args.overlap)

        print(f"\n{Colors.BOLD}Strategy:{Colors.RESET} {Colors.YELLOW}{args.strategy}{Colors.RESET}")
        print_success(f"Created {Colors.BOLD}{len(chunks)}{Colors.RESET} chunks")
        print(f"\n{Colors.BOLD}Chunks:{Colors.RESET}")
        print_chunks(chunks, args.verbose)

        if args.query:
            assert_ollama_up(args.host)
            print_header(f"Retrieval: \"{args.query}\"")

            results = retrieve_top_k(args.query, chunks, args.host, args.embed_model, args.top_k)
            for i, (chunk, score) in enumerate(results, 1):
                preview = chunk.text[:100].replace('\n', ' ')
                print(f"\n{Colors.CYAN}{i}.{Colors.RESET} [chunk {chunk.index}] score={Colors.GREEN}{score:.4f}{Colors.RESET}")
                print(f"   {Colors.DIM}\"{preview}...\"{Colors.RESET}")


if __name__ == "__main__":
    main(sys.argv[1:])
