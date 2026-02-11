#!/usr/bin/env python3
"""
llm_eval.py -- Rubric-based LLM evaluation using LLM-as-judge scoring

WHAT THIS IS
------------
A single-file script that evaluates an Ollama model's output quality across
5 categories (factual, reasoning, summarization, extraction, creative) using
a second model as an automated judge. Each task has a reference answer and
a rubric so the judge can assign a 1-10 score with reasoning.

WHY THIS EXISTS
---------------
LLM-as-judge is one of the most practical evaluation techniques: it scales
to hundreds of tasks, provides interpretable scores, and correlates well
with human preferences. This demo shows the full loop—generate, judge,
aggregate—so you can adapt it to your own eval datasets.

REQUIREMENTS
------------
- Python 3.9+
- `pip install requests`
- Ollama running locally: `ollama serve`
- Models pulled: `ollama pull llama3.2`

HOW TO RUN
----------
# Evaluate llama3.2 on all tasks
python llm_eval.py

# Compare two models
python llm_eval.py --models llama3.2 llama3.1

# Run a single category
python llm_eval.py --category reasoning --verbose

# Run a single task
python llm_eval.py --task factual_1 --verbose

# List all built-in tasks
python llm_eval.py --list-tasks
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
DEFAULT_MODEL = "llama3.2"
DEFAULT_JUDGE = "llama3.2"


# -----------------------------
# Data Structures
# -----------------------------
@dataclass
class EvalTask:
    """A single evaluation task with question, reference answer, and rubric."""
    task_id: str
    category: str
    question: str
    reference_answer: str
    rubric: str


@dataclass
class EvalResult:
    """Result of evaluating a single task."""
    task_id: str
    category: str
    response: str
    judge_score: int
    judge_reasoning: str
    response_time: float


@dataclass
class ModelReport:
    """Aggregated evaluation report for one model."""
    model: str
    avg_score: float
    category_scores: Dict[str, float]
    avg_response_time: float
    results: List[EvalResult] = field(default_factory=list)


# -----------------------------
# Built-in Evaluation Dataset
# -----------------------------
EVAL_TASKS: List[EvalTask] = [
    # --- Factual (2) ---
    EvalTask(
        task_id="factual_1",
        category="factual",
        question="What is the capital of Australia? Explain briefly why people often get this wrong.",
        reference_answer="The capital of Australia is Canberra, not Sydney or Melbourne. People often get this wrong because Sydney is the largest and most internationally recognized city, and Melbourne is another major city, but Canberra was purpose-built as the capital in 1913 as a compromise between the two rival cities.",
        rubric="Score 8-10 if the answer correctly states Canberra AND explains the common confusion with Sydney/Melbourne. Score 5-7 if correct capital but weak or missing explanation. Score 1-4 if wrong capital or major factual errors.",
    ),
    EvalTask(
        task_id="factual_2",
        category="factual",
        question="What is HTTP status code 418 and what is its origin?",
        reference_answer="HTTP 418 'I'm a teapot' was defined in RFC 2324 (Hyper Text Coffee Pot Control Protocol) as an April Fools' joke in 1998. It indicates the server is a teapot and refuses to brew coffee. Despite being a joke, it has been implemented in many real HTTP libraries and frameworks.",
        rubric="Score 8-10 if answer names 418 'I'm a teapot', mentions RFC 2324 or HTCPCP, and notes the April Fools'/joke origin. Score 5-7 if correct code meaning but missing RFC or joke context. Score 1-4 if wrong code meaning.",
    ),
    # --- Reasoning (2) ---
    EvalTask(
        task_id="reasoning_1",
        category="reasoning",
        question="A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
        reference_answer="The farmer has 9 sheep left. 'All but 9' means 9 remain. This is a language trick -- it does NOT mean 17 minus 9.",
        rubric="Score 8-10 if answer is 9 with clear explanation of the language trick. Score 5-7 if answer is 9 but poor explanation. Score 1-4 if answer is 8 or any other wrong number.",
    ),
    EvalTask(
        task_id="reasoning_2",
        category="reasoning",
        question="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        reference_answer="5 minutes. Each machine makes 1 widget in 5 minutes. So 100 machines each making 1 widget still takes 5 minutes, not 100 minutes. The rate is 1 widget per machine per 5 minutes.",
        rubric="Score 8-10 if answer is 5 minutes with clear rate-based reasoning. Score 5-7 if answer is 5 minutes but explanation is unclear. Score 1-4 if answer is wrong (e.g. 100 minutes, 1 minute).",
    ),
    # --- Summarization (2) ---
    EvalTask(
        task_id="summarization_1",
        category="summarization",
        question="Summarize the key innovation of the Transformer architecture in 2-3 sentences. Focus on what made it different from previous sequence models.",
        reference_answer="The Transformer replaced recurrent processing (RNNs/LSTMs) with self-attention, allowing it to process all positions in a sequence simultaneously rather than sequentially. This enabled massive parallelization during training and better capture of long-range dependencies. The architecture uses multi-head attention, positional encodings, and an encoder-decoder structure.",
        rubric="Score 8-10 if summary mentions self-attention replacing recurrence, parallelization benefit, and is 2-3 sentences. Score 5-7 if mentions attention but misses parallelization or is wrong length. Score 1-4 if factually wrong or misses the core innovation.",
    ),
    EvalTask(
        task_id="summarization_2",
        category="summarization",
        question="Summarize Marie Curie's main achievements in exactly one sentence.",
        reference_answer="Marie Curie was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences (Physics in 1903 for research on radiation, Chemistry in 1911 for discovering polonium and radium).",
        rubric="Score 8-10 if one sentence mentioning two Nobel Prizes in different fields. Score 5-7 if mentions Nobel Prize(s) but missing key details or not exactly one sentence. Score 1-4 if major factual errors or way too long.",
    ),
    # --- Extraction (2) ---
    EvalTask(
        task_id="extraction_1",
        category="extraction",
        question="Extract all dates and their associated events from this text:\n\n'The company was founded on March 15, 2010. It launched its first product on July 4, 2012. The IPO took place on November 22, 2018, and the merger was completed on January 8, 2023.'\n\nFormat as a list of date: event pairs.",
        reference_answer="- March 15, 2010: Company founded\n- July 4, 2012: First product launched\n- November 22, 2018: IPO took place\n- January 8, 2023: Merger completed",
        rubric="Score 8-10 if all 4 date-event pairs extracted correctly in a clear list format. Score 5-7 if 3 pairs correct or formatting is poor. Score 1-4 if 2 or fewer pairs correct or dates wrong.",
    ),
    EvalTask(
        task_id="extraction_2",
        category="extraction",
        question="Extract structured fields from this restaurant review:\n\n'Visited Bella Napoli last Friday. The margherita pizza was outstanding -- crispy thin crust with fresh mozzarella. Service was slow though, waited 40 minutes for our food. Overall 4 out of 5 stars. Price was reasonable at about $15 per person.'\n\nExtract: restaurant name, dish mentioned, positive aspects, negative aspects, rating, price range.",
        reference_answer="- Restaurant: Bella Napoli\n- Dish: Margherita pizza\n- Positive: Outstanding pizza, crispy thin crust, fresh mozzarella, reasonable price\n- Negative: Slow service, 40 minute wait\n- Rating: 4/5\n- Price: ~$15 per person",
        rubric="Score 8-10 if all 6 fields extracted correctly. Score 5-7 if 4-5 fields correct. Score 1-4 if 3 or fewer fields correct or major extraction errors.",
    ),
    # --- Creative (2) ---
    EvalTask(
        task_id="creative_1",
        category="creative",
        question="Write exactly 3 bullet points explaining why code review is important. Each bullet should be one sentence.",
        reference_answer="- Code review catches bugs early before they reach production, reducing the cost of fixes.\n- It spreads knowledge across the team so no single person is a bottleneck for any part of the codebase.\n- Reviews enforce consistent coding standards and encourage better design decisions.",
        rubric="Score 8-10 if exactly 3 bullets, each one sentence, covering distinct and valid reasons. Score 5-7 if 3 bullets but some are multi-sentence or reasons overlap. Score 1-4 if wrong number of bullets or reasons are invalid.",
    ),
    EvalTask(
        task_id="creative_2",
        category="creative",
        question="Write a brief, professional email (3-5 sentences) politely declining a meeting invitation because of a scheduling conflict. Do not use the word 'unfortunately'.",
        reference_answer="Subject: Re: Meeting Invitation\n\nThank you for including me in the upcoming meeting. I have a prior commitment at that time and won't be able to attend. I'd appreciate it if you could share the meeting notes or recording afterward so I can stay up to date. Please let me know if there's another time slot that works for a follow-up discussion.",
        rubric="Score 8-10 if email is professional, 3-5 sentences, declines due to conflict, does NOT use 'unfortunately'. Score 5-7 if mostly good but uses 'unfortunately' or wrong length. Score 1-4 if unprofessional, too long/short, or doesn't actually decline.",
    ),
]


# -----------------------------
# Utilities
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


def query_ollama(host: str, model: str, prompt: str, temperature: float = 0.3) -> str:
    """Send a prompt to Ollama and return the response."""
    resp = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 800},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# -----------------------------
# LLM Evaluator
# -----------------------------
class LLMEvaluator:
    """Evaluates LLM responses using a judge model."""

    JUDGE_PROMPT = """You are an expert evaluator. Score the following AI response on a scale of 1-10 based on the rubric provided.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

AI RESPONSE TO EVALUATE:
{response}

RUBRIC:
{rubric}

Provide your evaluation in this exact format:
SCORE: [1-10]
REASONING: [2-3 sentences explaining your score based on the rubric]"""

    def __init__(self, host: str, judge_model: str):
        self.host = host
        self.judge_model = judge_model

    def run_task(self, model: str, task: EvalTask, verbose: bool = False, show_full: bool = False) -> EvalResult:
        """Generate a response and judge it for a single task."""
        # Generate response from the model being evaluated
        t0 = time.time()
        response = query_ollama(self.host, model, task.question)
        response_time = time.time() - t0

        # Judge the response
        judge_prompt = self.JUDGE_PROMPT.format(
            question=task.question,
            reference=task.reference_answer,
            response=response,
            rubric=task.rubric,
        )
        judge_output = query_ollama(self.host, self.judge_model, judge_prompt, temperature=0.1)
        score, reasoning = self._parse_judge_output(judge_output)

        if show_full:
            print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}")
            print(f"{Colors.BOLD}Task:{Colors.RESET} {task.task_id} ({task.category})")
            print(f"{Colors.BOLD}Question:{Colors.RESET}\n{task.question}")
            print(f"\n{Colors.BOLD}Reference Answer:{Colors.RESET}\n{task.reference_answer}")
            print(f"\n{Colors.BOLD}Model Response ({model}):{Colors.RESET}\n{response}")
            print(f"\n{Colors.BOLD}Judge Output ({self.judge_model}):{Colors.RESET}\n{judge_output}")
            print(f"\n{Colors.BOLD}Parsed Score:{Colors.RESET} {score}/10")
            print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
        elif verbose:
            print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}")
            print(f"{Colors.BOLD}Task:{Colors.RESET} {task.task_id} ({task.category})")
            print(f"{Colors.BOLD}Question:{Colors.RESET} {task.question[:100]}...")
            print(f"{Colors.BOLD}Response:{Colors.RESET} {response[:200]}...")
            print(f"{Colors.BOLD}Score:{Colors.RESET} {score}/10")
            print(f"{Colors.BOLD}Reasoning:{Colors.RESET} {reasoning}")
            print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")

        return EvalResult(
            task_id=task.task_id,
            category=task.category,
            response=response,
            judge_score=score,
            judge_reasoning=reasoning,
            response_time=response_time,
        )

    def evaluate_model(
        self, model: str, tasks: List[EvalTask], verbose: bool = False, show_full: bool = False
    ) -> ModelReport:
        """Evaluate a model across all tasks and aggregate results."""
        results: List[EvalResult] = []

        for i, task in enumerate(tasks, 1):
            print_progress(
                f"Task {i}/{len(tasks)}: {Colors.BOLD}{task.task_id}{Colors.RESET} ({task.category})"
            )
            result = self.run_task(model, task, verbose=verbose, show_full=show_full)
            results.append(result)

            # Progress indicator
            score_color = Colors.GREEN if result.judge_score >= 7 else (
                Colors.YELLOW if result.judge_score >= 5 else Colors.RED
            )
            print(
                f"  {score_color}{result.judge_score}/10{Colors.RESET} "
                f"{Colors.DIM}({result.response_time:.1f}s){Colors.RESET}"
            )

        # Aggregate scores
        avg_score = sum(r.judge_score for r in results) / len(results) if results else 0
        avg_time = sum(r.response_time for r in results) / len(results) if results else 0

        # Per-category averages
        category_scores: Dict[str, List[int]] = {}
        for r in results:
            category_scores.setdefault(r.category, []).append(r.judge_score)
        cat_avgs = {cat: sum(scores) / len(scores) for cat, scores in category_scores.items()}

        return ModelReport(
            model=model,
            avg_score=avg_score,
            category_scores=cat_avgs,
            avg_response_time=avg_time,
            results=results,
        )

    def _parse_judge_output(self, text: str) -> tuple:
        """Parse SCORE and REASONING from judge output."""
        score = 5  # default
        reasoning = ""

        # Extract score
        score_match = re.search(r"SCORE:\s*(\d+)", text, re.IGNORECASE)
        if score_match:
            parsed = int(score_match.group(1))
            score = max(1, min(10, parsed))

        # Extract reasoning
        reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            # Take only the first paragraph to keep it concise
            reasoning = reasoning.split("\n\n")[0].strip()
        else:
            # Fallback: use everything after the score line
            reasoning = text.strip()

        return score, reasoning


# -----------------------------
# Display
# -----------------------------
def print_report(report: ModelReport) -> None:
    """Print a formatted evaluation report for one model."""
    print_header(f"Evaluation Report: {report.model}")

    # Overall
    score_color = Colors.GREEN if report.avg_score >= 7 else (
        Colors.YELLOW if report.avg_score >= 5 else Colors.RED
    )
    print(f"\n  {Colors.BOLD}Overall Score:{Colors.RESET} {score_color}{report.avg_score:.1f}/10{Colors.RESET}")
    print(f"  {Colors.BOLD}Avg Response Time:{Colors.RESET} {Colors.DIM}{report.avg_response_time:.1f}s{Colors.RESET}")

    # Per-category
    print(f"\n  {Colors.BOLD}{'Category':<18} {'Score':<12}{Colors.RESET}")
    print(f"  {Colors.DIM}{'-' * 30}{Colors.RESET}")
    for cat in sorted(report.category_scores):
        cat_score = report.category_scores[cat]
        cat_color = Colors.GREEN if cat_score >= 7 else (
            Colors.YELLOW if cat_score >= 5 else Colors.RED
        )
        print(f"  {cat:<18} {cat_color}{cat_score:.1f}/10{Colors.RESET}")
    print(f"  {Colors.DIM}{'-' * 30}{Colors.RESET}")


def print_comparison(reports: List[ModelReport]) -> None:
    """Print a side-by-side comparison table for multiple models."""
    print_header("Model Comparison")

    categories = sorted({cat for r in reports for cat in r.category_scores})

    # Header row
    model_names = [r.model for r in reports]
    header = f"  {Colors.BOLD}{'Category':<18}"
    for name in model_names:
        header += f" {name:<14}"
    header += Colors.RESET
    print(header)
    print(f"  {Colors.DIM}{'-' * (18 + 15 * len(model_names))}{Colors.RESET}")

    # Category rows
    for cat in categories:
        row = f"  {cat:<18}"
        for report in reports:
            score = report.category_scores.get(cat, 0)
            color = Colors.GREEN if score >= 7 else (Colors.YELLOW if score >= 5 else Colors.RED)
            row += f" {color}{score:<14.1f}{Colors.RESET}"
        print(row)

    # Overall row
    print(f"  {Colors.DIM}{'-' * (18 + 15 * len(model_names))}{Colors.RESET}")
    row = f"  {Colors.BOLD}{'OVERALL':<18}{Colors.RESET}"
    for report in reports:
        color = Colors.GREEN if report.avg_score >= 7 else (
            Colors.YELLOW if report.avg_score >= 5 else Colors.RED
        )
        row += f" {color}{Colors.BOLD}{report.avg_score:<14.1f}{Colors.RESET}"
    print(row)

    # Timing row
    row = f"  {Colors.DIM}{'Avg time':<18}"
    for report in reports:
        row += f" {report.avg_response_time:<14.1f}"
    row += Colors.RESET
    print(row)

    # Winner
    best = max(reports, key=lambda r: r.avg_score)
    if len(reports) > 1:
        print(f"\n  {Colors.GREEN}Best overall:{Colors.RESET} {Colors.BOLD}{best.model}{Colors.RESET} ({best.avg_score:.1f}/10)")


def list_tasks(tasks: List[EvalTask]) -> None:
    """Print all available evaluation tasks."""
    print_header("Available Evaluation Tasks")

    current_cat = ""
    for task in tasks:
        if task.category != current_cat:
            current_cat = task.category
            print(f"\n  {Colors.BOLD}{Colors.CYAN}{current_cat.upper()}{Colors.RESET}")

        print(f"    {Colors.BOLD}{task.task_id:<20}{Colors.RESET} {task.question[:60]}...")

    print(f"\n  {Colors.DIM}Total: {len(tasks)} tasks across {len(set(t.category for t in tasks))} categories{Colors.RESET}")


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rubric-based LLM evaluation using LLM-as-judge scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--models", nargs="+", default=[DEFAULT_MODEL],
        help="Model(s) to evaluate (space-separated for comparison)",
    )
    p.add_argument("--judge", default=DEFAULT_JUDGE, help="Model to use as judge")
    p.add_argument("--host", default=DEFAULT_HOST, help="Ollama host URL")
    p.add_argument(
        "--category",
        choices=sorted(set(t.category for t in EVAL_TASKS)),
        help="Run only tasks in this category",
    )
    p.add_argument("--task", help="Run a single task by ID")
    p.add_argument("--verbose", action="store_true", help="Show truncated responses and judge reasoning")
    p.add_argument("--show-full", action="store_true", help="Show full untruncated output from both the evaluated model and the judge")
    p.add_argument("--list-tasks", action="store_true", help="List all tasks and exit")
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    # --list-tasks
    if args.list_tasks:
        list_tasks(EVAL_TASKS)
        return

    # Filter tasks
    tasks = EVAL_TASKS
    if args.task:
        tasks = [t for t in tasks if t.task_id == args.task]
        if not tasks:
            print_error(f"Task '{args.task}' not found. Use --list-tasks to see available tasks.")
            return
    elif args.category:
        tasks = [t for t in tasks if t.category == args.category]

    print_header("LLM Evaluation")
    print(f"  {Colors.BOLD}Models:{Colors.RESET} {', '.join(args.models)}")
    print(f"  {Colors.BOLD}Judge:{Colors.RESET} {args.judge}")
    print(f"  {Colors.BOLD}Tasks:{Colors.RESET} {len(tasks)}")

    # Verify Ollama
    assert_ollama_up(args.host)

    evaluator = LLMEvaluator(host=args.host, judge_model=args.judge)
    reports: List[ModelReport] = []

    for model in args.models:
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Evaluating: {model}{Colors.RESET}")
        report = evaluator.evaluate_model(model, tasks, verbose=args.verbose, show_full=args.show_full)
        reports.append(report)
        print_report(report)

    # Comparison table for multiple models
    if len(reports) > 1:
        print_comparison(reports)


if __name__ == "__main__":
    main(sys.argv[1:])
