#!/usr/bin/env python3
"""
rag_eval.py -- Evaluate both retrieval quality and answer quality of a RAG pipeline

WHAT THIS IS
------------
A single-file script that builds a mini RAG pipeline over a fictional company's
engineering docs, then evaluates it with:
  - Retrieval metrics (precision@k, recall@k, MRR) -- computed algorithmically
  - Answer quality (faithfulness, relevance) -- scored by an LLM judge

You can compare different configurations (k values, chunk sizes) to see how
they affect both retrieval and answer quality.

WHY THIS EXISTS
---------------
Building a RAG pipeline is only half the work -- knowing whether it works well
is the other half. This demo shows how to measure retrieval and generation
quality systematically, which is essential before deploying RAG in production.

REQUIREMENTS
------------
- Python 3.9+
- `pip install requests`
- Ollama running locally: `ollama serve`
- Models pulled: `ollama pull llama3.2` and `ollama pull nomic-embed-text`

HOW TO RUN
----------
# Default evaluation (k=3, chunk_size=400)
python rag_eval.py

# Compare retrieval depths
python rag_eval.py --k 1 3 5

# Compare chunk sizes
python rag_eval.py --chunk-size 200 400 800

# Fast mode: retrieval metrics only (no LLM judge)
python rag_eval.py --skip-answer-eval

# Run a single question with full detail
python rag_eval.py --question qa_1 --verbose

# List all questions and ground truth
python rag_eval.py --list-questions
"""

import argparse
import math
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}")


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
DEFAULT_JUDGE = "llama3.2"
DEFAULT_TOP_K = 3
DEFAULT_CHUNK_SIZE = 400
EMBED_PAYLOAD_KEY = "prompt"


# -----------------------------
# Data Structures
# -----------------------------
@dataclass
class Document:
    """A knowledge base document."""
    doc_id: str
    title: str
    content: str


@dataclass
class QAExample:
    """A question with ground-truth relevant docs and expected answer."""
    qa_id: str
    question: str
    relevant_doc_ids: List[str]
    expected_answer: str


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics for a single query."""
    qa_id: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    retrieved_doc_ids: List[str]


@dataclass
class AnswerMetrics:
    """Answer quality metrics for a single query."""
    qa_id: str
    faithfulness: int
    relevance: int
    reasoning: str
    raw_judge_output: str = ""


@dataclass
class RAGEvalReport:
    """Aggregated evaluation report for one RAG configuration."""
    config_label: str
    k: int
    chunk_size: int
    avg_precision: float
    avg_recall: float
    avg_mrr: float
    avg_faithfulness: float
    avg_relevance: float
    retrieval_results: List[RetrievalMetrics] = field(default_factory=list)
    answer_results: List[AnswerMetrics] = field(default_factory=list)


# -----------------------------
# Knowledge Base (fictional company)
# -----------------------------
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
# QA Dataset
# -----------------------------
QA_EXAMPLES: List[QAExample] = [
    QAExample(
        qa_id="qa_1",
        question="How many code reviews are required before merging a pull request?",
        relevant_doc_ids=["doc2"],
        expected_answer="At least two approving reviews are required before merging.",
    ),
    QAExample(
        qa_id="qa_2",
        question="What database system is used in production and how often are backups taken?",
        relevant_doc_ids=["doc4"],
        expected_answer="Production uses PostgreSQL 15. Automated backups are taken every 6 hours with a 30-day retention policy.",
    ),
    QAExample(
        qa_id="qa_3",
        question="What are the severity levels for incidents and which ones require a postmortem?",
        relevant_doc_ids=["doc7"],
        expected_answer="There are four severity levels: SEV1 (service down), SEV2 (major feature broken), SEV3 (minor impact), SEV4 (cosmetic). SEV1 and SEV2 require blameless postmortems within 3 business days.",
    ),
    QAExample(
        qa_id="qa_4",
        question="What is the rate limit for API requests?",
        relevant_doc_ids=["doc5"],
        expected_answer="Rate limiting is enforced at 1000 requests per minute per API key.",
    ),
    QAExample(
        qa_id="qa_5",
        question="How does the monitoring system handle critical alerts?",
        relevant_doc_ids=["doc3"],
        expected_answer="Critical alerts page the on-call engineer via PagerDuty. On-call engineers must respond within 15 minutes. Critical alerts must have a linked runbook.",
    ),
    QAExample(
        qa_id="qa_6",
        question="What testing frameworks are used for end-to-end tests and when do they run?",
        relevant_doc_ids=["doc6"],
        expected_answer="End-to-end tests use Playwright and run against the staging environment nightly as part of the 'nightly' test suite.",
    ),
    QAExample(
        qa_id="qa_7",
        question="How does a code change go from pull request to production deployment?",
        relevant_doc_ids=["doc1", "doc8"],
        expected_answer="A pull request triggers the CI/CD pipeline (lint, unit tests, integration tests, preview). After merging, deployment uses blue-green with canary analysis: 5% traffic for 30 minutes, then gradual rollout to 25%, 50%, 100% over two hours.",
    ),
    QAExample(
        qa_id="qa_8",
        question="What are the requirements for database schema migrations?",
        relevant_doc_ids=["doc4"],
        expected_answer="Schema migrations use Alembic and must be backwards-compatible for zero-downtime deployments. All migrations are reviewed by the DBA team before execution.",
    ),
]


# -----------------------------
# Embedding and Retrieval Utilities
# -----------------------------
def assert_ollama_up(host: str) -> None:
    """Quick health check: hit /api/tags to verify the daemon is reachable."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
    except RequestException as e:
        raise SystemExit(
            f"[!] Could not reach Ollama at {host}. Is `ollama serve` running?\n{e}"
        )


def normalize_ws(text: str) -> str:
    """Collapse internal whitespace and trim ends."""
    return " ".join(text.split()).strip()


def chunk_text(text: str, max_chars: int, overlap: int = 50) -> List[str]:
    """Character-based chunker with overlap."""
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += max_chars - overlap
        if start < len(text) and len(text) - start < overlap:
            break
    return chunks


def embed_text(host: str, model: str, text: str) -> List[float]:
    """Embed a single string via Ollama /api/embeddings."""
    url = f"{host}/api/embeddings"
    try:
        resp = requests.post(
            url, json={"model": model, EMBED_PAYLOAD_KEY: text, "options": {}}, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        if "embedding" in data:
            return data["embedding"]
    except RequestException:
        pass

    # Fallback to alternate key
    alt_key = "input" if EMBED_PAYLOAD_KEY == "prompt" else "prompt"
    resp = requests.post(url, json={"model": model, alt_key: text, "options": {}}, timeout=60)
    resp.raise_for_status()
    return resp.json()["embedding"]


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)


def query_ollama(host: str, model: str, prompt: str, temperature: float = 0.3) -> str:
    """Send a prompt to Ollama and return the response."""
    resp = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 600},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# -----------------------------
# Mini RAG Pipeline
# -----------------------------
class MiniRAG:
    """A minimal RAG pipeline for evaluation."""

    def __init__(self, host: str, gen_model: str, emb_model: str, chunk_size: int):
        self.host = host
        self.gen_model = gen_model
        self.emb_model = emb_model
        self.chunk_size = chunk_size
        # index: list of (doc_id, chunk_text, embedding)
        self.index: List[Tuple[str, str, List[float]]] = []

    def build_index(self, documents: List[Document]) -> None:
        """Chunk and embed all documents."""
        self.index = []
        for doc in documents:
            chunks = chunk_text(doc.content, max_chars=self.chunk_size)
            for chunk in chunks:
                vec = embed_text(self.host, self.emb_model, chunk)
                self.index.append((doc.doc_id, chunk, vec))

    def retrieve(self, question: str, k: int) -> List[Tuple[float, str, str]]:
        """Retrieve top-k chunks. Returns [(score, doc_id, chunk_text)]."""
        q_vec = embed_text(self.host, self.emb_model, question)
        scored = [(cosine(q_vec, vec), doc_id, chunk) for doc_id, chunk, vec in self.index]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def generate_answer(self, question: str, retrieved: List[Tuple[float, str, str]]) -> str:
        """Generate a RAG answer from retrieved context."""
        context_parts = []
        for i, (score, doc_id, chunk) in enumerate(retrieved, 1):
            context_parts.append(f"[{i}] {chunk}")
        context = "\n\n".join(context_parts)

        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return query_ollama(self.host, self.gen_model, prompt)


# -----------------------------
# RAG Evaluator
# -----------------------------
class RAGEvaluator:
    """Evaluates a RAG pipeline on retrieval and answer quality."""

    JUDGE_PROMPT = """You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

AI-GENERATED ANSWER:
{answer}

EXPECTED ANSWER:
{expected}

Evaluate the AI answer on two dimensions:

1. FAITHFULNESS (1-10): Is the answer grounded in the retrieved context? Does it only state things supported by the context, without hallucinating?
2. RELEVANCE (1-10): Does the answer actually address the question? Is it complete and useful?

Respond in this exact format:
FAITHFULNESS: [1-10]
RELEVANCE: [1-10]
REASONING: [2-3 sentences explaining your scores]"""

    def __init__(self, host: str, judge_model: str):
        self.host = host
        self.judge_model = judge_model

    def compute_retrieval_metrics(
        self, qa: QAExample, retrieved: List[Tuple[float, str, str]]
    ) -> RetrievalMetrics:
        """Compute precision@k, recall@k, and MRR for a single query."""
        retrieved_doc_ids = [doc_id for _, doc_id, _ in retrieved]
        relevant_set = set(qa.relevant_doc_ids)

        # Precision@k: fraction of retrieved chunks that come from relevant docs
        relevant_hits = sum(1 for d in retrieved_doc_ids if d in relevant_set)
        precision = relevant_hits / len(retrieved_doc_ids) if retrieved_doc_ids else 0

        # Recall@k: fraction of relevant docs that appear in retrieved results
        found_relevant = relevant_set & set(retrieved_doc_ids)
        recall = len(found_relevant) / len(relevant_set) if relevant_set else 0

        # MRR: 1 / rank of first relevant chunk
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids, 1):
            if doc_id in relevant_set:
                mrr = 1.0 / rank
                break

        return RetrievalMetrics(
            qa_id=qa.qa_id,
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            retrieved_doc_ids=retrieved_doc_ids,
        )

    def judge_answer(
        self,
        qa: QAExample,
        answer: str,
        retrieved: List[Tuple[float, str, str]],
    ) -> AnswerMetrics:
        """Use LLM judge to score faithfulness and relevance."""
        context_parts = [f"[{i}] {chunk}" for i, (_, _, chunk) in enumerate(retrieved, 1)]
        context = "\n\n".join(context_parts)

        prompt = self.JUDGE_PROMPT.format(
            question=qa.question,
            context=context,
            answer=answer,
            expected=qa.expected_answer,
        )
        judge_output = query_ollama(self.host, self.judge_model, prompt, temperature=0.1)
        metrics = self._parse_judge_output(qa.qa_id, judge_output)
        metrics.raw_judge_output = judge_output
        return metrics

    def evaluate_config(
        self,
        rag: MiniRAG,
        qa_examples: List[QAExample],
        k: int,
        skip_answer_eval: bool = False,
        verbose: bool = False,
        show_full: bool = False,
    ) -> RAGEvalReport:
        """Evaluate one RAG configuration across all QA examples."""
        retrieval_results: List[RetrievalMetrics] = []
        answer_results: List[AnswerMetrics] = []

        for i, qa in enumerate(qa_examples, 1):
            print_progress(
                f"Question {i}/{len(qa_examples)}: {Colors.BOLD}{qa.qa_id}{Colors.RESET}"
            )

            # Retrieve
            retrieved = rag.retrieve(qa.question, k=k)
            ret_metrics = self.compute_retrieval_metrics(qa, retrieved)
            retrieval_results.append(ret_metrics)

            # Generate and judge answer
            ans_metrics = None
            if not skip_answer_eval:
                answer = rag.generate_answer(qa.question, retrieved)
                ans_metrics = self.judge_answer(qa, answer, retrieved)
                answer_results.append(ans_metrics)

            # Progress output
            p_color = Colors.GREEN if ret_metrics.precision_at_k >= 0.5 else Colors.RED
            r_color = Colors.GREEN if ret_metrics.recall_at_k >= 0.5 else Colors.RED
            line = (
                f"  P@{k}={p_color}{ret_metrics.precision_at_k:.2f}{Colors.RESET} "
                f"R@{k}={r_color}{ret_metrics.recall_at_k:.2f}{Colors.RESET} "
                f"MRR={ret_metrics.mrr:.2f}"
            )
            if ans_metrics:
                f_color = Colors.GREEN if ans_metrics.faithfulness >= 7 else (
                    Colors.YELLOW if ans_metrics.faithfulness >= 5 else Colors.RED
                )
                v_color = Colors.GREEN if ans_metrics.relevance >= 7 else (
                    Colors.YELLOW if ans_metrics.relevance >= 5 else Colors.RED
                )
                line += (
                    f" | Faith={f_color}{ans_metrics.faithfulness}/10{Colors.RESET} "
                    f"Rel={v_color}{ans_metrics.relevance}/10{Colors.RESET}"
                )
            print(line)

            if show_full:
                print(f"\n    {Colors.DIM}{'─' * 56}{Colors.RESET}")
                print(f"    {Colors.BOLD}Question:{Colors.RESET}\n    {qa.question}")
                print(f"    {Colors.BOLD}Expected Answer:{Colors.RESET}\n    {qa.expected_answer}")
                print(f"    {Colors.BOLD}Retrieved docs:{Colors.RESET} {ret_metrics.retrieved_doc_ids}  (expected: {qa.relevant_doc_ids})")
                for j, (score, doc_id, chunk) in enumerate(retrieved, 1):
                    print(f"    {Colors.BOLD}[{j}] {doc_id} (sim={score:.3f}):{Colors.RESET}\n    {chunk}")
                if not skip_answer_eval:
                    print(f"\n    {Colors.BOLD}RAG Answer ({rag.gen_model}):{Colors.RESET}\n    {answer}")
                    if ans_metrics:
                        print(f"\n    {Colors.BOLD}Judge Output ({self.judge_model}):{Colors.RESET}\n    {ans_metrics.raw_judge_output}")
                print(f"    {Colors.DIM}{'─' * 56}{Colors.RESET}")
            elif verbose:
                print(f"    {Colors.DIM}Retrieved: {ret_metrics.retrieved_doc_ids}{Colors.RESET}")
                print(f"    {Colors.DIM}Expected:  {qa.relevant_doc_ids}{Colors.RESET}")
                if not skip_answer_eval:
                    print(f"    {Colors.DIM}Answer: {answer[:150]}...{Colors.RESET}")
                    if ans_metrics:
                        print(f"    {Colors.DIM}Judge: {ans_metrics.reasoning}{Colors.RESET}")

        # Aggregate
        avg_p = sum(r.precision_at_k for r in retrieval_results) / len(retrieval_results)
        avg_r = sum(r.recall_at_k for r in retrieval_results) / len(retrieval_results)
        avg_mrr = sum(r.mrr for r in retrieval_results) / len(retrieval_results)
        avg_faith = (
            sum(a.faithfulness for a in answer_results) / len(answer_results)
            if answer_results else 0
        )
        avg_rel = (
            sum(a.relevance for a in answer_results) / len(answer_results)
            if answer_results else 0
        )

        return RAGEvalReport(
            config_label=f"k={k}, chunk={rag.chunk_size}",
            k=k,
            chunk_size=rag.chunk_size,
            avg_precision=avg_p,
            avg_recall=avg_r,
            avg_mrr=avg_mrr,
            avg_faithfulness=avg_faith,
            avg_relevance=avg_rel,
            retrieval_results=retrieval_results,
            answer_results=answer_results,
        )

    def _parse_judge_output(self, qa_id: str, text: str) -> AnswerMetrics:
        """Parse faithfulness, relevance, and reasoning from judge output."""
        faithfulness = 5
        relevance = 5
        reasoning = ""

        faith_match = re.search(r"FAITHFULNESS:\s*(\d+)", text, re.IGNORECASE)
        if faith_match:
            faithfulness = max(1, min(10, int(faith_match.group(1))))

        rel_match = re.search(r"RELEVANCE:\s*(\d+)", text, re.IGNORECASE)
        if rel_match:
            relevance = max(1, min(10, int(rel_match.group(1))))

        reason_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip().split("\n\n")[0].strip()
        else:
            reasoning = text.strip()

        return AnswerMetrics(
            qa_id=qa_id,
            faithfulness=faithfulness,
            relevance=relevance,
            reasoning=reasoning,
        )


# -----------------------------
# Display
# -----------------------------
def print_report(report: RAGEvalReport, skip_answer: bool = False) -> None:
    """Print a formatted report for one configuration."""
    print_header(f"RAG Eval Report: {report.config_label}")

    # Retrieval metrics
    print(f"\n  {Colors.BOLD}Retrieval Metrics{Colors.RESET}")
    print(f"  {Colors.DIM}{'-' * 40}{Colors.RESET}")

    p_color = Colors.GREEN if report.avg_precision >= 0.5 else Colors.RED
    r_color = Colors.GREEN if report.avg_recall >= 0.5 else Colors.RED
    m_color = Colors.GREEN if report.avg_mrr >= 0.5 else Colors.RED
    print(f"  Precision@{report.k}:  {p_color}{report.avg_precision:.3f}{Colors.RESET}")
    print(f"  Recall@{report.k}:     {r_color}{report.avg_recall:.3f}{Colors.RESET}")
    print(f"  MRR:          {m_color}{report.avg_mrr:.3f}{Colors.RESET}")

    if not skip_answer:
        print(f"\n  {Colors.BOLD}Answer Quality{Colors.RESET}")
        print(f"  {Colors.DIM}{'-' * 40}{Colors.RESET}")
        f_color = Colors.GREEN if report.avg_faithfulness >= 7 else (
            Colors.YELLOW if report.avg_faithfulness >= 5 else Colors.RED
        )
        r_color = Colors.GREEN if report.avg_relevance >= 7 else (
            Colors.YELLOW if report.avg_relevance >= 5 else Colors.RED
        )
        print(f"  Faithfulness:  {f_color}{report.avg_faithfulness:.1f}/10{Colors.RESET}")
        print(f"  Relevance:     {r_color}{report.avg_relevance:.1f}/10{Colors.RESET}")


def print_comparison(reports: List[RAGEvalReport], skip_answer: bool = False) -> None:
    """Print a comparison table across configurations."""
    print_header("Configuration Comparison")

    # Header
    header = f"  {Colors.BOLD}{'Config':<22} {'P@k':<8} {'R@k':<8} {'MRR':<8}"
    if not skip_answer:
        header += f" {'Faith':<8} {'Rel':<8}"
    header += Colors.RESET
    print(header)
    print(f"  {Colors.DIM}{'-' * (46 if skip_answer else 62)}{Colors.RESET}")

    for r in reports:
        p_color = Colors.GREEN if r.avg_precision >= 0.5 else Colors.RED
        rc_color = Colors.GREEN if r.avg_recall >= 0.5 else Colors.RED
        m_color = Colors.GREEN if r.avg_mrr >= 0.5 else Colors.RED
        row = (
            f"  {r.config_label:<22} "
            f"{p_color}{r.avg_precision:<8.3f}{Colors.RESET} "
            f"{rc_color}{r.avg_recall:<8.3f}{Colors.RESET} "
            f"{m_color}{r.avg_mrr:<8.3f}{Colors.RESET}"
        )
        if not skip_answer:
            f_color = Colors.GREEN if r.avg_faithfulness >= 7 else (
                Colors.YELLOW if r.avg_faithfulness >= 5 else Colors.RED
            )
            v_color = Colors.GREEN if r.avg_relevance >= 7 else (
                Colors.YELLOW if r.avg_relevance >= 5 else Colors.RED
            )
            row += (
                f" {f_color}{r.avg_faithfulness:<8.1f}{Colors.RESET} "
                f"{v_color}{r.avg_relevance:<8.1f}{Colors.RESET}"
            )
        print(row)

    print(f"  {Colors.DIM}{'-' * (46 if skip_answer else 62)}{Colors.RESET}")

    # Best config by recall
    best = max(reports, key=lambda r: r.avg_recall)
    print(f"\n  {Colors.GREEN}Best recall:{Colors.RESET} {Colors.BOLD}{best.config_label}{Colors.RESET} (R@k={best.avg_recall:.3f})")


def list_questions(qa_examples: List[QAExample]) -> None:
    """Print all QA examples with ground truth."""
    print_header("Available QA Examples")

    for qa in qa_examples:
        print(f"\n  {Colors.BOLD}{Colors.CYAN}{qa.qa_id}{Colors.RESET}")
        print(f"    Q: {qa.question}")
        print(f"    {Colors.DIM}Relevant docs: {qa.relevant_doc_ids}{Colors.RESET}")
        print(f"    {Colors.DIM}Expected: {qa.expected_answer[:80]}...{Colors.RESET}")

    print(f"\n  {Colors.DIM}Total: {len(qa_examples)} questions{Colors.RESET}")


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality and answer quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default=DEFAULT_HOST, help="Ollama host URL")
    p.add_argument(
        "--models", nargs="+", default=[DEFAULT_GEN_MODEL],
        help="Gen model(s) for RAG answers (space-separated to compare models)",
    )
    p.add_argument("--embed-model", default=DEFAULT_EMB_MODEL, help="Embedding model")
    p.add_argument("--judge", default=DEFAULT_JUDGE, help="Model to use as judge")
    p.add_argument(
        "--k", nargs="+", type=int, default=[DEFAULT_TOP_K],
        help="Top-k values to evaluate (space-separated for comparison)",
    )
    p.add_argument(
        "--chunk-size", nargs="+", type=int, default=[DEFAULT_CHUNK_SIZE],
        help="Chunk sizes to evaluate (space-separated for comparison)",
    )
    p.add_argument("--question", help="Run a single question by ID")
    p.add_argument("--skip-answer-eval", action="store_true", help="Skip LLM judge, retrieval metrics only")
    p.add_argument("--verbose", action="store_true", help="Show retrieved chunks and judge reasoning")
    p.add_argument("--show-full", action="store_true", help="Show full untruncated output from both the RAG model and the judge")
    p.add_argument("--list-questions", action="store_true", help="List all questions and exit")
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    # --list-questions
    if args.list_questions:
        list_questions(QA_EXAMPLES)
        return

    # Filter questions
    qa_examples = QA_EXAMPLES
    if args.question:
        qa_examples = [q for q in qa_examples if q.qa_id == args.question]
        if not qa_examples:
            print_error(f"Question '{args.question}' not found. Use --list-questions to see available questions.")
            return

    # Build all config combinations: (model, k, chunk_size)
    configs = [(m, k, cs) for m in args.models for k in args.k for cs in args.chunk_size]

    print_header("RAG Pipeline Evaluation")
    print(f"  {Colors.BOLD}Gen Model(s):{Colors.RESET} {', '.join(args.models)}")
    print(f"  {Colors.BOLD}Embed Model:{Colors.RESET} {args.embed_model}")
    print(f"  {Colors.BOLD}Judge:{Colors.RESET} {args.judge}")
    print(f"  {Colors.BOLD}Questions:{Colors.RESET} {len(qa_examples)}")
    print(f"  {Colors.BOLD}Configs:{Colors.RESET} {len(configs)} (models={args.models}, k={args.k}, chunk={args.chunk_size})")
    if args.skip_answer_eval:
        print(f"  {Colors.YELLOW}Skipping answer evaluation (retrieval metrics only){Colors.RESET}")

    # Verify Ollama
    assert_ollama_up(args.host)

    evaluator = RAGEvaluator(host=args.host, judge_model=args.judge)
    reports: List[RAGEvalReport] = []

    # Cache index per (chunk_size, embed_model) to avoid redundant embedding
    index_cache: Dict[int, MiniRAG] = {}

    for gen_model, k_val, chunk_val in configs:
        label = f"{gen_model}, k={k_val}, chunk={chunk_val}"
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Config: {label}{Colors.RESET}")

        # Build or reuse index for this chunk size
        if chunk_val not in index_cache:
            print_progress(f"Building index (chunk_size={chunk_val})...")
            t0 = time.time()
            rag = MiniRAG(args.host, gen_model, args.embed_model, chunk_val)
            rag.build_index(DOCUMENTS)
            index_cache[chunk_val] = rag
            print_success(
                f"Indexed {Colors.BOLD}{len(rag.index)}{Colors.RESET} chunks in {time.time() - t0:.1f}s"
            )
        else:
            rag = index_cache[chunk_val]
            # Update gen_model for this run (index is shared, only generation differs)
            rag.gen_model = gen_model
            print_success(f"Reusing cached index ({len(rag.index)} chunks)")

        # Evaluate
        report = evaluator.evaluate_config(
            rag, qa_examples, k=k_val,
            skip_answer_eval=args.skip_answer_eval,
            verbose=args.verbose,
            show_full=args.show_full,
        )
        # Override label to include model name when comparing models
        if len(args.models) > 1:
            report.config_label = label
        reports.append(report)
        print_report(report, skip_answer=args.skip_answer_eval)

    # Comparison table for multiple configs
    if len(reports) > 1:
        print_comparison(reports, skip_answer=args.skip_answer_eval)


if __name__ == "__main__":
    main(sys.argv[1:])
