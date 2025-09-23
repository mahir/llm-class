# llm-class

Sample code for the "Business Applications of Large Language Models" course (IEORE4573, Columbia University). The repository currently focuses on a simple retrieval-augmented generation (RAG) chatbot that runs entirely on your machine and demonstrates the full retrieval → augmentation → generation workflow with a local LLM served by [Ollama](https://ollama.com/).

## Project Layout
- `week3/simple_rag_demo/rag_demo.py` — end-to-end demo that vectorizes a small FAQ, retrieves relevant snippets with TF‑IDF cosine similarity, and asks an Ollama model to answer user questions with the retrieved context.

## Prerequisites
- Python 3.9+
- `pip install scikit-learn numpy requests`
- [Ollama](https://ollama.com/download) installed and running locally via `ollama serve`
- An Ollama model pulled to your machine (default: `ollama pull llama3.2`)

## Quick Start
1. (Optional) create and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
2. Install Python dependencies: `pip install scikit-learn numpy requests`
3. Start Ollama in another terminal: `ollama serve`
4. Pull the default model if needed: `ollama pull llama3.2`
5. Run the demo from the repo root: `python week3/simple_rag_demo/rag_demo.py`

The script prints sample questions, waits for your input, shows which FAQ entries it retrieved, and returns a grounded answer from the local model.

## How the Demo Works
1. **Indexing** — `SimpleRAG.add_documents` loads a list of FAQ articles and builds TF‑IDF vectors so queries and documents live in the same vector space.
2. **Retrieval** — `SimpleRAG.retrieve` converts your question into a TF‑IDF vector, scores every document with cosine similarity, and returns the top matches above a small relevance threshold.
3. **Prompt Assembly** — `SimpleRAG.query` formats the retrieved snippets into a context block together with instructions about staying factual.
4. **Generation** — `SimpleRAG.generate_with_ollama` calls the Ollama REST API with that prompt and streams the final answer back to the console.

The knowledge base is intentionally tiny and hard-coded in `create_sample_knowledge_base()` so you can focus on observing the RAG pipeline without extra setup.

## Customizing & Extending
- Swap in a different Ollama model by passing `SimpleRAG(ollama_model="model-name")` or editing the constructor call in `main()`.
- Replace `create_sample_knowledge_base()` with your own loader that reads markdown, PDFs, or database records—just return a list of dicts containing `title` and `content`.
- Tweak retrieval quality by adjusting TF‑IDF parameters (e.g., `ngram_range`, `min_df`) or by upgrading to embedding-based retrieval.
- Wrap `SimpleRAG.query` in a REST endpoint, Slack bot, or web UI to expose the same workflow outside the CLI.

## Troubleshooting
- **Missing packages** — the script checks imports on startup and reminds you to install dependencies if it exits early.
- **Model not found** — see the list of locally available models at `http://localhost:11434/api/tags` or pull a new one with `ollama pull <name>`.
- **Connection errors** — ensure `ollama serve` is running on the same machine and accessible at `http://localhost:11434`.

## License
This project is released under the MIT License. See `LICENSE` for details.
