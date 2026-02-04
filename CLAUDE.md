# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational Python repository for Columbia's "Business Applications of Large Language Models" course (IEORE4573). Demonstrates practical LLM patterns using local Ollama models and OpenAI APIs, with focus on RAG, structured output, and model evaluation.

## Common Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify Ollama is running
curl http://localhost:11434/api/tags

# Run demos (from repo root)
python week3/simple-rag/simple-rag.py
python week3/simple-rag-v2/mini_rag_ollama.py "Your question here"
python week3/arxiv-summarizer/arxiv-summarizer.py 2103.00020 --type technical
python week3/image-processor/image-processor.py week3/image-processor/images -m "llava:7b"
python week3/prompting-techniques/prompting_techniques.py --technique all
python week4/structured-output/structured_output.py
python week4/llm-judge/ollama_judge.py --model-a llama3.1 --model-b qwen2:7b
python week4/openai-api/test_multiple.py  # Requires OPENAI_API_KEY
```

## Architecture

**Week 3 - Local Ollama Workflows:**
- `simple-rag/` - TF-IDF based RAG with `SimpleRAG` class (index → retrieve → augment → generate)
- `simple-rag-v2/` - Dense vector RAG using Ollama embeddings and cosine similarity
- `arxiv-summarizer/` - PDF extraction + multi-style summarization
- `image-processor/` - Vision model integration (llava)
- `simple-batch/` - Batch processing patterns
- `spanish-tutor-ollama/` - Custom Ollama model via Modelfile
- `prompting-techniques/` - Compare 8 prompting strategies (zero-shot, few-shot, CoT, self-consistency, role, step-back, least-to-most)

**Week 4 - Evaluation & Structured Output:**
- `structured-output/` - JSON schema enforcement via Ollama's `format: "json"`
- `llm-judge/` - Model comparison with judge evaluation and score parsing
- `openai-api/` - OpenAI patterns with JSON schema validation and retry logic

## External Services

**Ollama (localhost:11434):**
- Endpoints: `/api/generate`, `/api/chat`, `/api/embeddings`, `/api/tags`
- Models used: `llama3.2`, `llama3.1`, `nomic-embed-text`, `qwen3:32b`, `llava:7b`

**OpenAI:**
- Models: `gpt-4.1`, `gpt-4o-mini`, `gpt-5-nano`
- Requires `OPENAI_API_KEY` environment variable

## Code Patterns

- Scripts are single-file and self-contained (intentional for teaching)
- Knowledge bases are hardcoded inline for simplicity
- Uses dataclasses for structured data in newer modules
- Output includes ASCII emoji for visual feedback
