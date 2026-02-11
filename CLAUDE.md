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
python week3/01-simple-batch/simple-batch.py
python week3/02-prompting-techniques/prompting_techniques.py --technique all
python week3/03-tfidf-rag/simple-rag.py
python week3/04-embedding-rag/mini_rag_ollama.py "Your question here"
python week3/07-arxiv-summarizer/arxiv-summarizer.py 2103.00020 --type technical
python week3/08-image-processor/image-processor.py week3/08-image-processor/images -m "llava:7b"
python week4/01-structured-output/structured_output.py
python week4/02-llm-judge/ollama_judge.py --model-a llama3.2 --model-b llama3.1
python week4/03-openai-api/openai_structured.py  # Requires OPENAI_API_KEY
python week4/04-llm-eval/llm_eval.py                    # Rubric-based eval
python week4/04-llm-eval/llm_eval.py --models llama3.2 llama3.1  # Compare models
python week4/05-rag-eval/rag_eval.py                     # RAG pipeline eval
python week4/05-rag-eval/rag_eval.py --k 1 3 5           # Compare retrieval depths
python week4/06-mcp-server/server.py                     # MCP server (stdio)
```

## Architecture

**Week 3 - Local Ollama Workflows (numbered in learning order):**
- `01-simple-batch/` - Batch processing patterns with Ollama API
- `02-prompting-techniques/` - Compare 8 prompting strategies (zero-shot, few-shot, CoT, self-consistency, role, step-back, least-to-most)
- `03-tfidf-rag/` - TF-IDF based RAG with `SimpleRAG` class (index → retrieve → augment → generate)
- `04-embedding-rag/` - Dense vector RAG using Ollama embeddings and cosine similarity
- `05-semantic-chunking/` - Compare 4 chunking strategies (character, sentence, paragraph, section)
- `06-hybrid-rag/` - Combined keyword + dense retrieval with Reciprocal Rank Fusion
- `07-arxiv-summarizer/` - PDF extraction + multi-style summarization
- `08-image-processor/` - Vision model integration (llava)
- `09-fastapi-rag/` - FastAPI REST endpoint with LlamaIndex + Ollama
- `10-ollama-models/` - Custom Ollama models via Modelfiles (spanish-tutor, socratic-tutor, personalities)

**Week 4 - Evaluation & Structured Output:**
- `01-structured-output/` - JSON schema enforcement via Ollama's `format: "json"`
- `02-llm-judge/` - Model comparison with judge evaluation and score parsing
- `03-openai-api/` - OpenAI structured output with JSON schema validation and retry logic
- `04-llm-eval/` - Rubric-based LLM evaluation with LLM-as-judge (10 tasks, 5 categories)
- `05-rag-eval/` - RAG pipeline evaluation: retrieval metrics (precision, recall, MRR) + answer quality (faithfulness, relevance)
- `06-mcp-server/` - MCP server exposing engineering docs as tools for Claude Desktop/Claude Code (TF-IDF search, no Ollama needed)

## External Services

**Ollama (localhost:11434):**
- Endpoints: `/api/generate`, `/api/chat`, `/api/embeddings`, `/api/tags`
- Models used: `llama3.2`, `llama3.1`, `nomic-embed-text`, `llava:7b`

**OpenAI:**
- Models: `gpt-4o-2024-08-06`, `gpt-4o-mini`
- Requires `OPENAI_API_KEY` environment variable

## Code Patterns

- Scripts are single-file and self-contained (intentional for teaching)
- Knowledge bases are hardcoded inline for simplicity
- Uses dataclasses for structured data in newer modules
- Output includes ASCII emoji for visual feedback
