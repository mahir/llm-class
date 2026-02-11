#!/usr/bin/env python3
"""
server.py -- MCP server exposing engineering docs as tools for Claude

WHAT THIS IS
------------
A Model Context Protocol (MCP) server that lets Claude Desktop (or Claude Code)
search and retrieve documents from a fictional company's engineering knowledge
base. It exposes three tools:

  - search_docs(query, k)  -- semantic search via TF-IDF, returns top-k matches
  - get_document(doc_id)   -- fetch a single document by ID
  - list_documents()       -- list all available document titles and IDs

The knowledge base is the same 8 engineering documents used in
week4/05-rag-eval/rag_eval.py, so you can compare Claude's answers (using MCP
tools) against the Ollama RAG pipeline evaluated there.

WHY THIS EXISTS
---------------
MCP is how Claude connects to external tools and data sources. This demo shows
how to build an MCP server from scratch using FastMCP -- the simplest way to
give Claude access to your own data. No Ollama dependency required; the server
uses TF-IDF for retrieval and starts instantly.

REQUIREMENTS
------------
- Python 3.10+ (required by the mcp package)
- `pip install mcp scikit-learn`

HOW TO RUN
----------
# Standalone (starts stdio server, Ctrl-C to stop)
python week4/06-mcp-server/server.py

# With Claude Desktop -- add this to your Claude Desktop config
# (Settings > Developer > Edit Config):
{
  "mcpServers": {
    "engineering-docs": {
      "command": "python",
      "args": ["/absolute/path/to/week4/06-mcp-server/server.py"]
    }
  }
}

# With Claude Code -- add via:
#   claude mcp add engineering-docs python /absolute/path/to/week4/06-mcp-server/server.py

Then ask Claude things like:
  - "List the available engineering documents"
  - "What's our code review policy?"
  - "How does the deployment process work?"
  - "What is the incident response process?"
"""

from dataclasses import dataclass
from typing import List

from mcp.server.fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Knowledge Base (fictional company -- same 8 docs as rag_eval.py)
# -----------------------------
@dataclass
class Document:
    """A knowledge base document."""
    doc_id: str
    title: str
    content: str


DOCUMENTS: List[Document] = [
    Document(
        doc_id="doc1",
        title="CI/CD Pipeline",
        content=(
            "Our CI/CD pipeline runs on every pull request. The pipeline has four stages: "
            "lint, unit tests, integration tests, and deployment preview. Linting uses flake8 "
            "and black for Python code, and eslint for JavaScript. Unit tests must maintain at "
            "least 85% code coverage. Integration tests run against a staging database that is "
            "reset before each run. Deployment previews create an ephemeral environment that "
            "stays active for 48 hours. The pipeline typically completes in 12-15 minutes. "
            "Failed pipelines block merging into the main branch. Pipeline configuration is "
            "stored in .github/workflows/ and changes to it require approval from the DevOps team."
        ),
    ),
    Document(
        doc_id="doc2",
        title="Code Review Process",
        content=(
            "All code changes require at least two approving reviews before merging. Reviewers "
            "should focus on correctness, readability, test coverage, and security implications. "
            "Reviews should be completed within one business day. Authors must respond to all "
            "comments before merging, even if the response is 'acknowledged'. Large PRs (over "
            "400 lines changed) should be broken into smaller, reviewable chunks. We use a "
            "CODEOWNERS file to automatically assign reviewers based on file paths. Review "
            "comments should be constructive and reference specific lines of code. If a reviewer "
            "is unsure about a change, they should request a meeting rather than blocking the PR."
        ),
    ),
    Document(
        doc_id="doc3",
        title="Monitoring and Alerting",
        content=(
            "We use Prometheus for metrics collection and Grafana for dashboards. Key metrics "
            "include request latency (p50, p95, p99), error rate, throughput, and resource "
            "utilization (CPU, memory, disk). Alerts are configured in PagerDuty with three "
            "severity levels: critical (pages on-call), warning (Slack notification), and info "
            "(logged only). Critical alerts must have a runbook linked in the alert definition. "
            "On-call engineers respond to critical alerts within 15 minutes. We conduct weekly "
            "reviews of alert fatigue and tune thresholds quarterly. Dashboards are organized "
            "by service, and each service must have a 'golden signals' dashboard."
        ),
    ),
    Document(
        doc_id="doc4",
        title="Database Management",
        content=(
            "Production databases run PostgreSQL 15 with read replicas for scaling read-heavy "
            "workloads. Schema migrations use Alembic and must be backwards-compatible to "
            "support zero-downtime deployments. All migrations are reviewed by the DBA team "
            "before execution. We take automated backups every 6 hours with a 30-day retention "
            "policy. Point-in-time recovery is available for the last 7 days. Connection pooling "
            "is handled by PgBouncer with a maximum of 200 connections per service. Slow queries "
            "(over 500ms) are logged and reviewed weekly. Database access in production requires "
            "a JIT (just-in-time) access request approved by a team lead."
        ),
    ),
    Document(
        doc_id="doc5",
        title="API Design Standards",
        content=(
            "All APIs follow REST conventions with JSON request and response bodies. Endpoints "
            "use plural nouns (e.g., /api/v2/users) and HTTP methods for actions. Versioning "
            "uses URL path prefixes (/api/v1/, /api/v2/). All endpoints require authentication "
            "via Bearer tokens issued by our OAuth2 provider. Rate limiting is enforced at 1000 "
            "requests per minute per API key. Responses include standard pagination using limit "
            "and offset parameters. Error responses follow RFC 7807 (Problem Details for HTTP "
            "APIs). Breaking changes require a new API version and a 6-month deprecation period. "
            "All new endpoints must have OpenAPI documentation before deployment."
        ),
    ),
    Document(
        doc_id="doc6",
        title="Testing Strategy",
        content=(
            "Our testing pyramid emphasizes unit tests at the base, integration tests in the "
            "middle, and a small number of end-to-end tests at the top. Unit tests should be "
            "fast (under 100ms each) and test a single behavior. Integration tests verify "
            "interactions between components, especially database queries and API calls. "
            "End-to-end tests use Playwright and run against the staging environment nightly. "
            "We maintain separate test suites: 'fast' (unit only, runs in CI), 'full' (unit + "
            "integration, runs pre-merge), and 'nightly' (all tests including e2e). Test data "
            "factories use the factory_boy library. Flaky tests are quarantined and tracked in "
            "a dedicated dashboard."
        ),
    ),
    Document(
        doc_id="doc7",
        title="Incident Response",
        content=(
            "Incidents are classified into four severity levels: SEV1 (service down), SEV2 "
            "(major feature broken), SEV3 (minor impact), SEV4 (cosmetic issue). SEV1 and "
            "SEV2 incidents trigger an immediate war room in Slack (#incident-active). The "
            "on-call engineer becomes the Incident Commander and coordinates response. All "
            "actions taken during an incident are logged in the incident channel. Post-incident "
            "reviews (blameless postmortems) are required for SEV1 and SEV2 within 3 business "
            "days. The postmortem must include a timeline, root cause analysis, impact assessment, "
            "and action items with owners and deadlines. Incident metrics (MTTR, MTTD, frequency) "
            "are tracked monthly."
        ),
    ),
    Document(
        doc_id="doc8",
        title="Deployment Process",
        content=(
            "Deployments follow a blue-green strategy with automated canary analysis. New code "
            "is first deployed to the canary environment (5% of traffic) for 30 minutes. If "
            "error rates stay below 0.1% and latency remains within 10% of baseline, traffic "
            "gradually shifts to 25%, 50%, and then 100% over two hours. Rollbacks are "
            "automatic if canary metrics exceed thresholds. Production deployments happen "
            "Monday through Thursday, with a freeze on Fridays and before holidays. Emergency "
            "hotfixes can bypass the freeze with VP approval. Deployment artifacts are Docker "
            "images stored in our private registry with semantic versioning."
        ),
    ),
]


# -----------------------------
# TF-IDF Index (built once at import time)
# -----------------------------
_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
_doc_vectors = _vectorizer.fit_transform([d.content for d in DOCUMENTS])
_doc_lookup = {d.doc_id: d for d in DOCUMENTS}


# -----------------------------
# MCP Server
# -----------------------------
mcp = FastMCP("engineering-docs")


@mcp.tool()
def search_docs(query: str, k: int = 3) -> str:
    """Search the engineering knowledge base for documents matching a query.

    Uses TF-IDF similarity to find the most relevant documents.

    Args:
        query: The search query (natural language question or keywords).
        k: Number of results to return (1-8, default 3).
    """
    k = max(1, min(k, len(DOCUMENTS)))

    query_vector = _vectorizer.transform([query])
    scores = cosine_similarity(query_vector, _doc_vectors).flatten()

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for idx, score in ranked:
        doc = DOCUMENTS[idx]
        results.append(
            f"[{doc.doc_id}] {doc.title} (relevance: {score:.3f})\n{doc.content}"
        )

    return "\n\n---\n\n".join(results)


@mcp.tool()
def get_document(doc_id: str) -> str:
    """Fetch a specific document by its ID.

    Use list_documents() first to see available IDs, or get an ID from
    search_docs() results.

    Args:
        doc_id: The document identifier (e.g. "doc1", "doc2", ..., "doc8").
    """
    doc = _doc_lookup.get(doc_id)
    if doc is None:
        available = ", ".join(sorted(_doc_lookup.keys()))
        return f"Document '{doc_id}' not found. Available IDs: {available}"
    return f"[{doc.doc_id}] {doc.title}\n\n{doc.content}"


@mcp.tool()
def list_documents() -> str:
    """List all documents in the engineering knowledge base.

    Returns document IDs and titles so you can decide which to retrieve.
    """
    lines = [f"  {d.doc_id}: {d.title}" for d in DOCUMENTS]
    return f"Available documents ({len(DOCUMENTS)} total):\n" + "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
