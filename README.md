# llm-class

Sample code for the "Business Applications of Large Language Models" course (IEORE4573, Columbia University). The repository now covers local Ollama RAG demos, structured output workflows, and OpenAI-powered evaluation tooling so you can explore end-to-end retrieval → augmentation → generation plus downstream analysis.

## Table of Contents
- [Project Layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [How the RAG Demo Works](#how-the-rag-demo-works)
- [Additional Demos](#additional-demos)
- [Customizing & Extending](#customizing--extending)
- [OpenAI API Setup](#openai-api-setup)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Layout
The repository is split into weekly modules so you can focus on one capability at a time. Directories are numbered in suggested learning order.

| Week | Focus | Key Scripts |
| --- | --- | --- |
| Week 3 | Local Ollama workflows (retrieval, image, batching, prompting) | `week3/01-simple-batch/`, `week3/02-prompting-techniques/`, `week3/03-tfidf-rag/`, `week3/04-embedding-rag/`, `week3/05-semantic-chunking/`, `week3/06-hybrid-rag/`, `week3/07-arxiv-summarizer/`, `week3/08-image-processor/`, `week3/09-fastapi-rag/`, `week3/10-ollama-models/` |
| Week 4 | Evaluation & structured output (Ollama + OpenAI) | `week4/01-structured-output/`, `week4/02-llm-judge/`, `week4/03-openai-api/`, `week4/04-llm-eval/`, `week4/05-rag-eval/`, `week4/06-mcp-server/` |

Each directory also includes helper modules and cached artifacts that demonstrate common data pipelines (chunking, embedding, evaluation logs, etc.).

## Prerequisites
- Python 3.9+
- `pip install -r requirements.txt`
- [Ollama](https://ollama.com/download) installed and running locally via `ollama serve`
- An Ollama model pulled to your machine (default: `ollama pull llama3.2`)
- All week4 dependencies (openai, jsonschema) are included in `requirements.txt`

## Quick Start
1. (Optional) create and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
2. Install Python dependencies: `pip install -r requirements.txt`
3. (Optional) verify Ollama connectivity: `curl http://localhost:11434/api/tags`
4. Start Ollama in another terminal: `ollama serve`
5. Pull the default model if needed: `ollama pull llama3.2`
6. Run the default RAG demo from the repo root: `python week3/03-tfidf-rag/simple-rag.py`

The script prints sample questions, waits for your input, shows which FAQ entries it retrieved, and returns a grounded answer from the local model.

### Verifying Your Environment
- Confirm Python dependencies resolved cleanly: `python -m pip check`
- Ensure you can reach the Ollama REST API before starting a script: `curl http://localhost:11434/api/generate -d '{"model":"llama3.2","prompt":"ping"}'`
- When using the OpenAI API examples, double-check that `OPENAI_API_KEY` is exported in the same shell session that runs the scripts.

## How the RAG Demo Works
1. **Indexing** — `SimpleRAG.add_documents` loads a list of FAQ articles and builds TF‑IDF vectors so queries and documents live in the same vector space.
2. **Retrieval** — `SimpleRAG.retrieve` converts your question into a TF‑IDF vector, scores every document with cosine similarity, and returns the top matches above a small relevance threshold.
3. **Prompt Assembly** — `SimpleRAG.query` formats the retrieved snippets into a context block together with instructions about staying factual.
4. **Generation** — `SimpleRAG.generate_with_ollama` calls the Ollama REST API with that prompt and streams the final answer back to the console.

The knowledge base is intentionally tiny and hard-coded in `create_sample_knowledge_base()` so you can focus on observing the RAG pipeline without extra setup.

## Additional Demos
- **Simple Batch**: `python week3/01-simple-batch/simple-batch.py`
  - Sends a series of tagging prompts to an Ollama model and writes a timestamped JSON report.
- **Prompting Techniques**: `python week3/02-prompting-techniques/prompting_techniques.py --technique all`
  - Compares 8 prompting strategies on challenging math/reasoning problems; use `--show-prompts` to see exactly what's sent to the LLM and `--list-models` to see available Ollama models.
- **TF-IDF RAG**: `python week3/03-tfidf-rag/simple-rag.py`
  - Interactive RAG chatbot using TF-IDF retrieval over a hardcoded FAQ knowledge base.
- **Embedding RAG**: `python week3/04-embedding-rag/mini_rag_ollama.py "How do embeddings help a RAG system?"`
  - Builds dense vector embeddings with Ollama, computes cosine similarity manually, and prompts a chat model using only retrieved context.
- **Semantic Chunking**: `python week3/05-semantic-chunking/semantic_chunker.py`
  - Compares 4 chunking strategies (character, sentence, paragraph, section) and their impact on retrieval quality.
- **Hybrid RAG**: `python week3/06-hybrid-rag/hybrid_rag.py`
  - Combines keyword (TF-IDF) and dense (embedding) retrieval using Reciprocal Rank Fusion.
- **ArXiv Summarizer**: `python week3/07-arxiv-summarizer/arxiv-summarizer.py 2103.00020 --type technical`
  - Downloads PDFs, caches them under `week3/07-arxiv-summarizer/arxiv_cache/`, extracts text, and produces multiple summary styles.
- **Image Processor**: `python week3/08-image-processor/image-processor.py week3/08-image-processor/images -m "llava:7b"`
  - Iterates through images, collects metadata via Pillow, and asks a vision-capable model for descriptions; results are saved to JSON.
- **FastAPI RAG**: `python week3/09-fastapi-rag/app.py`
  - Serves a RAG pipeline as a REST API using FastAPI + LlamaIndex + Ollama.
- **Ollama Models**: `week3/10-ollama-models/`
  - Custom Ollama model personalities via Modelfiles (Spanish tutor, Socratic tutor, and 3 personality archetypes).
- **Structured Output (Ollama)**: `python week4/01-structured-output/structured_output.py`
  - Forces JSON-only answers for entity extraction, planning, and complex business analysis—progresses from simple to deeply nested schemas.
- **Ollama Judge**: `python week4/02-llm-judge/ollama_judge.py --model-a llama3.2 --model-b llama3.1`
  - Compares outputs from two models and requests a third model to score and explain the winning answer; ideal for rapid regression testing.
- **Support Ticket Triage**: `python week4/03-openai-api/openai_structured.py`
  - Validates JSON-formatted answers against a schema and retries until the model produces well-formed tickets.
- **LLM Evaluation**: `python week4/04-llm-eval/llm_eval.py`
  - Rubric-based evaluation of LLM output quality across 5 categories (factual, reasoning, summarization, extraction, creative) using LLM-as-judge scoring; supports multi-model comparison.
- **RAG Evaluation**: `python week4/05-rag-eval/rag_eval.py`
  - Evaluates RAG retrieval quality (precision@k, recall@k, MRR) and answer quality (faithfulness, relevance) over a fictional company knowledge base; compare k values and chunk sizes.
- **MCP Server**: `python week4/06-mcp-server/server.py`
  - An MCP (Model Context Protocol) server that exposes the same engineering docs as searchable tools for Claude Desktop or Claude Code. Uses TF-IDF retrieval with no Ollama dependency.

### Suggested Learning Path

**Week 3 — Local Ollama Workflows (follow the numbered directories):**
1. `01-simple-batch/` — Learn the Ollama API basics (send prompt, get response)
2. `02-prompting-techniques/` — See how prompt engineering affects output quality
3. `03-tfidf-rag/` — Understand TF-IDF retrieval-augmented generation
4. `04-embedding-rag/` — Upgrade to dense vector embeddings for retrieval
5. `05-semantic-chunking/` — Compare chunking strategies and their impact on RAG
6. `06-hybrid-rag/` — Combine keyword + dense retrieval for best results
7. `07-arxiv-summarizer/` — Apply RAG to a real-world task (research papers)
8. `08-image-processor/` — Work with vision models (llava)
9. `09-fastapi-rag/` — Serve a RAG pipeline as a REST API
10. `10-ollama-models/` — Build custom model personalities via Modelfiles

**Week 4 — Evaluation & Structured Output:**
1. `01-structured-output/` — Enforce JSON responses from Ollama (simple → complex)
2. `02-llm-judge/` — Compare two models using a third as judge
3. `03-openai-api/` — OpenAI structured output with schema validation and retry logic
4. `04-llm-eval/` — Rubric-based LLM evaluation with LLM-as-judge (5 categories, 10 tasks)
5. `05-rag-eval/` — RAG pipeline evaluation: retrieval metrics + answer quality scoring
6. `06-mcp-server/` — Build an MCP server so Claude can search your docs directly

## Customizing & Extending
- Swap in different Ollama models by editing the constructor arguments (e.g., `SimpleRAG(ollama_model="model-name")`) or passing CLI flags such as `--model`.
- Replace `create_sample_knowledge_base()` with your own loader that reads markdown, PDFs, or database records—just return a list of dicts containing `title` and `content`.
- Tweak retrieval quality by adjusting TF-IDF parameters (e.g., `ngram_range`, `min_df`) or by upgrading to embedding-based retrieval.
- Adapt the CLI scripts into your own workflows (REST endpoints, scheduled batch jobs, UI integrations) by reusing the underlying classes.

## OpenAI API Setup
- Set `OPENAI_API_KEY` in your shell (`export OPENAI_API_KEY="sk-..."`) before running anything in `week4/03-openai-api/`.
- Optional but recommended: set `OPENAI_BASE_URL` if you are proxying requests through a gateway or Azure OpenAI deployment.
- Pick lightweight models (`gpt-4o-mini`, `gpt-5-nano`, etc.) if you want faster iteration; adjust the script defaults as needed.
- The evaluation and ticket-triage scripts write JSON artifacts next to the source so you can diff results across runs. Clean up old results if you want a fresh run.

## Troubleshooting
- **Missing packages** — each script prints friendly install hints if an import fails on startup.
- **Model not found** — see the list of locally available models at `http://localhost:11434/api/tags` or pull a new one with `ollama pull <name>`.
- **Connection errors** — ensure `ollama serve` is running on the same machine and accessible at `http://localhost:11434`.

## License
This project is released under the MIT License. See `LICENSE` for details.
