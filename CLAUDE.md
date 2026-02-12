# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational Python repository for Columbia's "Business Applications of Large Language Models" course (IEORE4573). Demonstrates practical LLM patterns using local Ollama models and OpenAI APIs, with focus on RAG, structured output, and model evaluation.

## Setup & Common Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify Ollama is running
curl http://localhost:11434/api/tags

# Run any demo from repo root (all scripts are standalone)
python week3/03-tfidf-rag/simple-rag.py
python week3/02-prompting-techniques/prompting_techniques.py --technique all
python week3/04-embedding-rag/mini_rag_ollama.py "Your question here"
python week4/04-llm-eval/llm_eval.py --models llama3.2 llama3.1
python week4/05-rag-eval/rag_eval.py --k 1 3 5

# FastAPI RAG (separate deps: llama-index, fastapi, uvicorn)
python week3/09-fastapi-rag/app.py

# MCP server (Python 3.10+ required)
python week4/06-mcp-server/server.py

# OpenAI scripts (requires OPENAI_API_KEY)
python week4/03-openai-api/openai_structured.py
```

There is no test suite — scripts are verified by running them directly.

## Architecture

Weekly modules, numbered in suggested learning order within each week:

**Week 3 — Local Ollama Workflows:**
- `01-simple-batch/` — Batch processing patterns with Ollama `/api/generate`
- `02-prompting-techniques/` — Compare 8 prompting strategies (zero-shot, few-shot, CoT, self-consistency, role, step-back, least-to-most)
- `03-tfidf-rag/` — TF-IDF RAG with `SimpleRAG` class (index → retrieve → augment → generate)
- `04-embedding-rag/` — Dense vector RAG using Ollama `/api/embeddings` + cosine similarity
- `05-semantic-chunking/` — Compare 4 chunking strategies (character, sentence, paragraph, section)
- `06-hybrid-rag/` — Keyword + dense retrieval with Reciprocal Rank Fusion
- `07-arxiv-summarizer/` — PDF extraction + multi-style summarization (caches PDFs in `arxiv_cache/`)
- `08-image-processor/` — Vision model integration (llava)
- `09-fastapi-rag/` — FastAPI REST endpoint with LlamaIndex + Ollama (has its own deps)
- `10-ollama-models/` — Custom Ollama models via Modelfiles (spanish-tutor, socratic-tutor, personalities)
- `11-trivia-quiz/` — Interactive trivia game: multi-turn `/api/chat`, system prompts, JSON mode for structured back-and-forth

**Week 4 — Evaluation & Structured Output:**
- `01-structured-output/` — JSON schema enforcement via Ollama `format: "json"`
- `02-llm-judge/` — Model comparison with judge evaluation and score parsing
- `03-openai-api/` — OpenAI structured output with JSON schema validation and retry logic (uses `python-dotenv` for `.env` loading)
- `04-llm-eval/` — Rubric-based LLM evaluation with LLM-as-judge (10 tasks, 5 categories)
- `05-rag-eval/` — RAG pipeline evaluation: retrieval metrics (precision, recall, MRR) + answer quality
- `06-mcp-server/` — MCP server exposing engineering docs as tools for Claude Desktop/Code (TF-IDF search, no Ollama needed, Python 3.10+)

## Code Patterns

- **Single-file, self-contained scripts** — intentional for teaching. Each script includes all its logic, knowledge bases, and prompts inline. No shared library modules.
- **`Colors` class** — duplicated in 10 scripts; provides ANSI terminal colors with auto-disable when stdout is not a tty.
- **Ollama interaction** — direct `requests.post()` to `http://localhost:11434/api/{generate,chat,embeddings}`; no SDK. Most scripts accept `--host` / `--model` CLI flags via `argparse`.
- **Dataclasses** — used for structured data in newer modules (`llm_eval.py`, `rag_eval.py`, `ollama_judge.py`, `openai_structured.py`, `server.py`).
- **OpenAI usage** — only `week4/03-openai-api/` uses the OpenAI SDK; loads keys via `python-dotenv` from local `.env` or repo root `.env`.

## External Services

**Ollama (localhost:11434):**
- Endpoints: `/api/generate`, `/api/chat`, `/api/embeddings`, `/api/tags`
- Models: `llama3.2`, `llama3.1`, `nomic-embed-text`, `llava:7b`

**OpenAI (week4/03-openai-api only):**
- Requires `OPENAI_API_KEY` env var
- Models configured: `gpt-5`, `gpt-4o-mini`, `gpt-4.1-nano`
