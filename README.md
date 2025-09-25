# llm-class

Sample code for the "Business Applications of Large Language Models" course (IEORE4573, Columbia University). The repository now covers local Ollama RAG demos, structured output workflows, and OpenAI-powered evaluation tooling so you can explore end-to-end retrieval → augmentation → generation plus downstream analysis.

## Project Layout
### Week 3 — Local Ollama Workflows
- `week3/simple-rag/simple-rag.py` — end-to-end FAQ chatbot that demonstrates the RAG loop (TF-IDF retrieval + Ollama-powered generation).
- `week3/simple-rag-v2/mini_rag_ollama.py` — single-file dense RAG walkthrough that calls `/api/embeddings`, scores chunks with cosine similarity, and asks an Ollama chat model for grounded answers.
- `week3/arxiv-summarizer/arxiv-summarizer.py` — downloads arXiv papers, chunks their text, and produces different summary styles with Ollama.
- `week3/image-processor/image-processor.py` — batch-describes local images with vision models such as `llava`.
- `week3/simple-batch/simple-batch.py` — minimal batch-processing example that extracts product tags across multiple prompts.
- `week3/spanish-tutor-ollama/Modelfile` — custom `ollama create` recipe that turns `mistral` into a supportive Spanish instructor.

### Week 4 — Evaluation & Structured Output
- `week4/structured-output/structured_output.py` — enforces JSON-only answers from Ollama with lightweight schema hints.
- `week4/structured-output/structured_output_complex.py` — expands the same pattern for multi-layer business scenario planning.
- `week4/openai-api/eval2.py` — OpenAI Responses API evaluator with retry logic, per-class metrics, and JSON logging.
- `week4/openai-api/eval.py` — simpler Chat Completions evaluator you can contrast with the Responses API flow.
- `week4/openai-api/test.py` and `test_multiple.py` — structured-output support ticket triage that validates responses against a JSON schema before accepting them.
- `week4/llm-judge/ollama_judge.py` — compares answers from two Ollama-served models and asks a judge model to pick a winner with rationale and scoring.

## Prerequisites
- Python 3.9+
- `pip install -r requirements.txt`
- [Ollama](https://ollama.com/download) installed and running locally via `ollama serve`
- An Ollama model pulled to your machine (default: `ollama pull llama3.2`)
- `pip install openai jsonschema` (needed for the week4 `openai-api/` workflows)

## Quick Start
1. (Optional) create and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
2. Install Python dependencies: `pip install -r requirements.txt`
3. Start Ollama in another terminal: `ollama serve`
4. Pull the default model if needed: `ollama pull llama3.2`
5. Run the default RAG demo from the repo root: `python week3/simple-rag/simple-rag.py`

The script prints sample questions, waits for your input, shows which FAQ entries it retrieved, and returns a grounded answer from the local model.

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
- **OpenAI Sentiment Eval**: `python week4/openai-api/eval2.py`
  - Runs an automated evaluation loop against a toy sentiment dataset; set `OPENAI_API_KEY` before executing.
- **Support Ticket Triage**: `python week4/openai-api/test_multiple.py`
  - Validates JSON-formatted answers against a schema and retries until the model produces well-formed tickets.
- **Model Comparator**: `python week4/llm-judge/ollama_judge.py --model-a llama3.1 --model-b qwen2:7b`
  - Generates responses from two local models, then asks a judge model to score and pick a winner with reasoning.

## Customizing & Extending
- Swap in different Ollama models by editing the constructor arguments (e.g., `SimpleRAG(ollama_model="model-name")`) or passing CLI flags such as `--model`.
- Replace `create_sample_knowledge_base()` with your own loader that reads markdown, PDFs, or database records—just return a list of dicts containing `title` and `content`.
- Tweak retrieval quality by adjusting TF-IDF parameters (e.g., `ngram_range`, `min_df`) or by upgrading to embedding-based retrieval.
- Adapt the CLI scripts into your own workflows (REST endpoints, scheduled batch jobs, UI integrations) by reusing the underlying classes.

## OpenAI API Setup
- Set `OPENAI_API_KEY` in your shell (`export OPENAI_API_KEY="sk-..."`) before running anything in `week4/openai-api/`.
- Pick lightweight models (`gpt-4o-mini`, `gpt-5-nano`, etc.) if you want faster iteration; adjust the script defaults as needed.
- The evaluation and ticket-triage scripts write JSON artifacts next to the source so you can diff results across runs.

## Troubleshooting
- **Missing packages** — each script prints friendly install hints if an import fails on startup.
- **Model not found** — see the list of locally available models at `http://localhost:11434/api/tags` or pull a new one with `ollama pull <name>`.
- **Connection errors** — ensure `ollama serve` is running on the same machine and accessible at `http://localhost:11434`.

## License
This project is released under the MIT License. See `LICENSE` for details.
