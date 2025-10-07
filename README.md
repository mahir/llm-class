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
The repository is split into weekly modules so you can focus on one capability at a time. The table below highlights the primary entry points.

| Week | Focus | Key Scripts |
| --- | --- | --- |
| Week 3 | Local Ollama workflows (retrieval, image, batching) | `week3/simple-rag/simple-rag.py`, `week3/simple-rag-v2/mini_rag_ollama.py`, `week3/arxiv-summarizer/arxiv-summarizer.py`, `week3/image-processor/image-processor.py`, `week3/simple-batch/simple-batch.py`, `week3/spanish-tutor-ollama/Modelfile` |
| Week 4 | Evaluation & structured output (Ollama + OpenAI) | `week4/structured-output/structured_output.py`, `week4/structured-output/structured_output_complex.py`, `week4/openai-api/eval.py`, `week4/openai-api/eval2.py`, `week4/openai-api/test.py`, `week4/openai-api/test_multiple.py`, `week4/llm-judge/ollama_judge.py` |

Each directory also includes helper modules and cached artifacts that demonstrate common data pipelines (chunking, embedding, evaluation logs, etc.).

## Prerequisites
- Python 3.9+
- `pip install -r requirements.txt`
- [Ollama](https://ollama.com/download) installed and running locally via `ollama serve`
- An Ollama model pulled to your machine (default: `ollama pull llama3.2`)
- `pip install openai jsonschema` (needed for the week4 `openai-api/` workflows)

## Quick Start
1. (Optional) create and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
2. Install Python dependencies: `pip install -r requirements.txt`
3. (Optional) verify Ollama connectivity: `curl http://localhost:11434/api/tags`
4. Start Ollama in another terminal: `ollama serve`
5. Pull the default model if needed: `ollama pull llama3.2`
6. Run the default RAG demo from the repo root: `python week3/simple-rag/simple-rag.py`

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
- **ArXiv Summarizer**: `python week3/arxiv-summarizer/arxiv-summarizer.py 2103.00020 --type technical`
  - Downloads PDFs, caches them under `week3/arxiv-summarizer/arxiv_cache/`, extracts text, and produces multiple summary styles (feedparser is bundled in `requirements.txt` for search mode).
- **Image Processor**: `python week3/image-processor/image-processor.py week3/image-processor/images -m "llava:7b"`
  - Iterates through images, collects metadata via Pillow, and asks a vision-capable model for descriptions; results are saved to JSON.
- **Simple Batch**: `python week3/simple-batch/simple-batch.py`
  - Sends a series of tagging prompts to an Ollama model and writes a timestamped JSON report.
- **Mini Dense RAG**: `python week3/simple-rag-v2/mini_rag_ollama.py "How do embeddings help a RAG system?"`
  - Builds embeddings with Ollama, computes cosine similarity manually, and prompts a chat model using only retrieved context.
- **Structured Output (Ollama)**: `python week4/structured-output/structured_output.py`
  - Forces JSON-only answers for quick entity extraction or planning tasks; swap to the complex version for multi-layer business reports.
- **Ollama Judge**: `python week4/llm-judge/ollama_judge.py --model-a llama3.1 --model-b qwen2:7b`
  - Compares outputs from two models and requests a third model to score and explain the winning answer; ideal for rapid regression testing.
- **OpenAI Sentiment Eval**: `python week4/openai-api/eval2.py`
  - Runs an automated evaluation loop against a toy sentiment dataset; set `OPENAI_API_KEY` before executing.
- **Support Ticket Triage**: `python week4/openai-api/test_multiple.py`
  - Validates JSON-formatted answers against a schema and retries until the model produces well-formed tickets.

### Suggested Learning Path
1. Run the Week 3 simple RAG demo to understand the baseline retrieval loop.
2. Experiment with the dense RAG variant or the ArXiv summarizer to explore more advanced retrieval options.
3. Move to Week 4 structured output scripts to practice enforcing JSON responses and schema validation.
4. Finish by benchmarking or judging models with the Ollama or OpenAI evaluation workflows.

## Customizing & Extending
- Swap in different Ollama models by editing the constructor arguments (e.g., `SimpleRAG(ollama_model="model-name")`) or passing CLI flags such as `--model`.
- Replace `create_sample_knowledge_base()` with your own loader that reads markdown, PDFs, or database records—just return a list of dicts containing `title` and `content`.
- Tweak retrieval quality by adjusting TF-IDF parameters (e.g., `ngram_range`, `min_df`) or by upgrading to embedding-based retrieval.
- Adapt the CLI scripts into your own workflows (REST endpoints, scheduled batch jobs, UI integrations) by reusing the underlying classes.

## OpenAI API Setup
- Set `OPENAI_API_KEY` in your shell (`export OPENAI_API_KEY="sk-..."`) before running anything in `week4/openai-api/`.
- Optional but recommended: set `OPENAI_BASE_URL` if you are proxying requests through a gateway or Azure OpenAI deployment.
- Pick lightweight models (`gpt-4o-mini`, `gpt-5-nano`, etc.) if you want faster iteration; adjust the script defaults as needed.
- The evaluation and ticket-triage scripts write JSON artifacts next to the source so you can diff results across runs. Clean up old results if you want a fresh run.

## Troubleshooting
- **Missing packages** — each script prints friendly install hints if an import fails on startup.
- **Model not found** — see the list of locally available models at `http://localhost:11434/api/tags` or pull a new one with `ollama pull <name>`.
- **Connection errors** — ensure `ollama serve` is running on the same machine and accessible at `http://localhost:11434`.

## License
This project is released under the MIT License. See `LICENSE` for details.
