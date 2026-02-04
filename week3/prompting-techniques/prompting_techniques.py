#!/usr/bin/env python3
"""
prompting_techniques.py — Compare Zero-Shot, Few-Shot, Chain-of-Thought, and Self-Consistency

WHAT THIS IS
------------
A single-file demo showing how different prompting strategies affect LLM accuracy
on math and reasoning problems. Compares:
  1) Zero-shot: Just ask the question
  2) Few-shot: Provide worked examples first
  3) Chain-of-thought (CoT): Ask the model to reason step-by-step
  4) Self-consistency: Sample multiple CoT responses, take majority vote
  5) Role: Add expert persona to prompt
  6) Step-back: Ask general question first, then answer specific
  7) Least-to-most: Break into subproblems, solve sequentially

WHY THIS EXISTS
---------------
Prompting technique can dramatically improve accuracy on reasoning tasks—often
more than switching models. This demo lets you see the effect firsthand.

REQUIREMENTS
------------
- Python 3.9+
- `pip install requests`
- Ollama running locally: `ollama serve`
- A model pulled: `ollama pull llama3.2`

HOW TO RUN
----------
# Run all techniques on default problems
python prompting_techniques.py

# Compare specific technique
python prompting_techniques.py --technique few-shot

# Use different model
python prompting_techniques.py --model llama3.1

# Adjust self-consistency samples
python prompting_techniques.py --technique self-consistency --samples 5

# Run with higher temperature for more diversity
python prompting_techniques.py --temperature 0.7

# See what prompts are being sent to the LLM
python prompting_techniques.py --technique cot --show-prompts

# List available Ollama models
python prompting_techniques.py --list-models
"""

import argparse
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple
import re
import requests
from requests.exceptions import RequestException


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_TEMPERATURE = 0.0  # Deterministic for reproducibility
SELF_CONSISTENCY_SAMPLES = 5


# -----------------------------
# Test Problems (with known answers)
# -----------------------------
# These problems are designed to be challenging and differentiate prompting techniques.
# They include distractors, multi-step reasoning, and common misconception traps.
PROBLEMS = [
    {
        "id": "distractor_1",
        "question": "A store has 15 employees. Each employee works 8 hours per day, 5 days a week. The store is open 7 days a week and sells an average of 240 items per day. How many hours does ONE employee work per week?",
        "answer": 40,
        "type": "distractor"
    },
    {
        "id": "backwards_1",
        "question": "After giving away 30% of her stickers, Maria has 35 stickers left. How many stickers did she have originally?",
        "answer": 50,
        "type": "backwards_reasoning"
    },
    {
        "id": "multi_step_1",
        "question": "A tank is being filled with water. It starts empty. In the first hour, 100 gallons are added. In the second hour, 50 gallons leak out. In the third hour, the amount of water doubles. In the fourth hour, 75 gallons are added. How many gallons are in the tank?",
        "answer": 175,
        "type": "multi_step"
    },
    {
        "id": "careful_reading",
        "question": "Tom is 5 years older than Jane. Jane is twice as old as Sam. If Sam is 8 years old, what is the sum of all three of their ages?",
        "answer": 45,
        "type": "careful_reading"
    },
    {
        "id": "rate_change",
        "question": "A car travels from City A to City B at 60 mph, then returns from City B to City A at 40 mph. The distance between the cities is 120 miles. What is the average speed for the entire round trip in mph?",
        "answer": 48,
        "type": "rate_problem"
    },
    {
        "id": "sequence_trap",
        "question": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
        "answer": 9,
        "type": "language_trap"
    },
    {
        "id": "nested_percent",
        "question": "A shirt originally costs $80. It goes on sale for 25% off. Then, an additional 10% discount is applied to the sale price. What is the final price in dollars?",
        "answer": 54,
        "type": "nested_calculation"
    },
    {
        "id": "work_problem",
        "question": "Alice can paint a room in 6 hours. Bob can paint the same room in 3 hours. If they work together, how many hours will it take them to paint the room?",
        "answer": 2,
        "type": "work_rate"
    },
]


# -----------------------------
# Few-Shot Examples (different from test problems)
# -----------------------------
FEW_SHOT_EXAMPLES = [
    {
        "question": "A farmer has 12 chickens. Each chicken lays 3 eggs per day. How many eggs are collected in one day?",
        "answer": "The farmer has 12 chickens, and each lays 3 eggs. So 12 x 3 = 36 eggs per day.\n\nThe answer is 36."
    },
    {
        "question": "If a pizza costs $15 and you want to split it equally among 3 people, how much does each person pay?",
        "answer": "The pizza costs $15 total. Splitting among 3 people: 15 / 3 = $5 each.\n\nThe answer is 5."
    },
    {
        "question": "A car uses 4 gallons of gas to drive 100 miles. How many gallons does it need to drive 250 miles?",
        "answer": "The car uses 4 gallons per 100 miles. For 250 miles: (250 / 100) x 4 = 2.5 x 4 = 10 gallons.\n\nThe answer is 10."
    },
]


# -----------------------------
# Prompt Templates
# -----------------------------
ZERO_SHOT_TEMPLATE = """{question}

Provide only the numerical answer."""


FEW_SHOT_TEMPLATE = """Here are some example math problems and their solutions:

{examples}

Now solve this problem:
{question}

Provide only the numerical answer."""


COT_TEMPLATE = """{question}

Let's think step by step to solve this problem. Show your reasoning, then provide the final numerical answer on a new line starting with "The answer is"."""


FEW_SHOT_COT_TEMPLATE = """Here are some example math problems solved step by step:

{examples}

Now solve this problem step by step:
{question}

Show your reasoning, then provide the final numerical answer on a new line starting with "The answer is"."""


ROLE_TEMPLATE = """You are an expert mathematician and problem solver with years of experience teaching math. You are known for your accuracy and clear explanations.

{question}

Provide only the numerical answer."""


STEP_BACK_TEMPLATE = """First, let's identify what type of problem this is and what general approach we should use.

Question: {question}

Step 1 - What type of problem is this and what's the general approach?
Step 2 - Now apply that approach to solve it.
Step 3 - State the final numerical answer on a line starting with "The answer is"."""


LEAST_TO_MOST_TEMPLATE = """Let's break this problem into smaller subproblems and solve each one.

Question: {question}

First, identify the subproblems:
Then solve each subproblem in order:
Finally, combine to get the final answer.
State the final numerical answer on a line starting with "The answer is"."""


# -----------------------------
# Utilities
# -----------------------------
def print_prompt(prompt: str, technique: str) -> None:
    """Print prompt with visual formatting."""
    print(f"\n{'─'*60}")
    print(f"PROMPT ({technique}):")
    print('─'*60)
    print(prompt)
    print('─'*60 + "\n")


def list_available_models(host: str) -> None:
    """Query Ollama for available models and print them."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        if not models:
            print("No models found. Pull a model with: ollama pull llama3.2")
            return
        print("Available Ollama models:")
        for m in models:
            size = m.get("size", 0)
            size_gb = size / (1024**3) if size else 0
            print(f"  - {m['name']:<30} ({size_gb:.1f} GB)")
    except RequestException as e:
        print(f"[!] Could not reach Ollama at {host}. Is `ollama serve` running?\n{e}")


def assert_ollama_up(host: str) -> None:
    """Verify Ollama is reachable."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
    except RequestException as e:
        raise SystemExit(f"[!] Could not reach Ollama at {host}. Is `ollama serve` running?\n{e}")


def query_ollama(
    host: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    timeout: int = 60
) -> str:
    """Send a prompt to Ollama and return the response."""
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 500,
        }
    }

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def extract_number(text: str) -> Optional[int]:
    """Extract the final numerical answer from model response."""
    # Look for "The answer is X" pattern first
    match = re.search(r"[Tt]he answer is[:\s]*(\d+)", text)
    if match:
        return int(match.group(1))

    # Look for "= X" at end of calculation
    match = re.search(r"=\s*(\d+)\s*$", text, re.MULTILINE)
    if match:
        return int(match.group(1))

    # Find all numbers and take the last one
    numbers = re.findall(r"\b(\d+)\b", text)
    if numbers:
        return int(numbers[-1])

    return None


def format_few_shot_examples(examples: List[Dict], include_reasoning: bool = False) -> str:
    """Format few-shot examples into a string."""
    formatted = []
    for i, ex in enumerate(examples, 1):
        if include_reasoning:
            formatted.append(f"Example {i}:\nQuestion: {ex['question']}\nSolution: {ex['answer']}")
        else:
            # Extract just the final number for non-CoT
            answer_num = extract_number(ex['answer'])
            formatted.append(f"Example {i}:\nQuestion: {ex['question']}\nAnswer: {answer_num}")
    return "\n\n".join(formatted)


# -----------------------------
# Prompting Techniques
# -----------------------------
def zero_shot(
    host: str, model: str, question: str, temperature: float, show_prompts: bool = False
) -> Tuple[str, Optional[int]]:
    """Zero-shot: Just ask the question directly."""
    prompt = ZERO_SHOT_TEMPLATE.format(question=question)
    if show_prompts:
        print_prompt(prompt, "zero-shot")
    response = query_ollama(host, model, prompt, temperature)
    answer = extract_number(response)
    return response, answer


def few_shot(
    host: str, model: str, question: str, temperature: float, show_prompts: bool = False
) -> Tuple[str, Optional[int]]:
    """Few-shot: Provide examples before the question."""
    examples = format_few_shot_examples(FEW_SHOT_EXAMPLES, include_reasoning=False)
    prompt = FEW_SHOT_TEMPLATE.format(examples=examples, question=question)
    if show_prompts:
        print_prompt(prompt, "few-shot")
    response = query_ollama(host, model, prompt, temperature)
    answer = extract_number(response)
    return response, answer


def chain_of_thought(
    host: str, model: str, question: str, temperature: float, show_prompts: bool = False
) -> Tuple[str, Optional[int]]:
    """Chain-of-thought: Ask model to reason step-by-step."""
    prompt = COT_TEMPLATE.format(question=question)
    if show_prompts:
        print_prompt(prompt, "cot")
    response = query_ollama(host, model, prompt, temperature)
    answer = extract_number(response)
    return response, answer


def few_shot_cot(
    host: str, model: str, question: str, temperature: float, show_prompts: bool = False
) -> Tuple[str, Optional[int]]:
    """Few-shot Chain-of-thought: Examples with reasoning + step-by-step."""
    examples = format_few_shot_examples(FEW_SHOT_EXAMPLES, include_reasoning=True)
    prompt = FEW_SHOT_COT_TEMPLATE.format(examples=examples, question=question)
    if show_prompts:
        print_prompt(prompt, "few-shot-cot")
    response = query_ollama(host, model, prompt, temperature)
    answer = extract_number(response)
    return response, answer


def self_consistency(
    host: str,
    model: str,
    question: str,
    temperature: float,
    num_samples: int = 5,
    show_prompts: bool = False
) -> Tuple[str, Optional[int]]:
    """
    Self-consistency: Sample multiple CoT responses, take majority vote.
    Uses higher temperature to get diverse reasoning paths.
    """
    # Use higher temperature for diversity
    sample_temp = max(temperature, 0.7)

    prompt = COT_TEMPLATE.format(question=question)
    if show_prompts:
        print_prompt(prompt, "self-consistency")

    answers = []
    responses = []

    for i in range(num_samples):
        response = query_ollama(host, model, prompt, sample_temp)
        responses.append(response)
        answer = extract_number(response)
        if answer is not None:
            answers.append(answer)

    # Majority vote
    if not answers:
        return "\n---\n".join(responses), None

    vote_counts = Counter(answers)
    majority_answer, count = vote_counts.most_common(1)[0]

    summary = f"Sampled {num_samples} responses, got answers: {answers}\n"
    summary += f"Majority vote ({count}/{num_samples}): {majority_answer}"

    return summary, majority_answer


def role_prompting(
    host: str, model: str, question: str, temperature: float, show_prompts: bool = False
) -> Tuple[str, Optional[int]]:
    """Role: Add expert persona to the prompt."""
    prompt = ROLE_TEMPLATE.format(question=question)
    if show_prompts:
        print_prompt(prompt, "role")
    response = query_ollama(host, model, prompt, temperature)
    answer = extract_number(response)
    return response, answer


def step_back(
    host: str, model: str, question: str, temperature: float, show_prompts: bool = False
) -> Tuple[str, Optional[int]]:
    """Step-back: First identify problem type, then solve."""
    prompt = STEP_BACK_TEMPLATE.format(question=question)
    if show_prompts:
        print_prompt(prompt, "step-back")
    response = query_ollama(host, model, prompt, temperature)
    answer = extract_number(response)
    return response, answer


def least_to_most(
    host: str, model: str, question: str, temperature: float, show_prompts: bool = False
) -> Tuple[str, Optional[int]]:
    """Least-to-most: Break into subproblems and solve sequentially."""
    prompt = LEAST_TO_MOST_TEMPLATE.format(question=question)
    if show_prompts:
        print_prompt(prompt, "least-to-most")
    response = query_ollama(host, model, prompt, temperature)
    answer = extract_number(response)
    return response, answer


# -----------------------------
# Evaluation
# -----------------------------
TECHNIQUES = {
    "zero-shot": zero_shot,
    "few-shot": few_shot,
    "cot": chain_of_thought,
    "few-shot-cot": few_shot_cot,
    "self-consistency": self_consistency,
    "role": role_prompting,
    "step-back": step_back,
    "least-to-most": least_to_most,
}


def evaluate_technique(
    host: str,
    model: str,
    technique_name: str,
    problems: List[Dict],
    temperature: float,
    num_samples: int = 5,
    verbose: bool = False,
    show_prompts: bool = False
) -> Dict:
    """Evaluate a technique on all problems."""
    technique_fn = TECHNIQUES[technique_name]

    correct = 0
    results = []

    for problem in problems:
        question = problem["question"]
        expected = problem["answer"]

        # Run the technique
        if technique_name == "self-consistency":
            response, predicted = technique_fn(
                host, model, question, temperature, num_samples, show_prompts
            )
        else:
            response, predicted = technique_fn(
                host, model, question, temperature, show_prompts
            )

        is_correct = predicted == expected
        if is_correct:
            correct += 1

        result = {
            "id": problem["id"],
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        }

        if verbose:
            result["response"] = response

        results.append(result)

        # Progress indicator
        status = "[/] correct" if is_correct else "[X] wrong"
        print(f"  {problem['id']}: expected={expected}, got={predicted} {status}")

    accuracy = correct / len(problems) if problems else 0

    return {
        "technique": technique_name,
        "correct": correct,
        "total": len(problems),
        "accuracy": accuracy,
        "results": results
    }


def run_comparison(
    host: str,
    model: str,
    techniques: List[str],
    problems: List[Dict],
    temperature: float,
    num_samples: int,
    verbose: bool,
    show_prompts: bool = False
) -> List[Dict]:
    """Run all specified techniques and compare results."""
    all_results = []

    for technique in techniques:
        print(f"\n{'='*60}")
        print(f"Running: {technique.upper()}")
        print('='*60)

        t0 = time.time()
        result = evaluate_technique(
            host, model, technique, problems, temperature, num_samples, verbose, show_prompts
        )
        elapsed = time.time() - t0

        result["elapsed_seconds"] = round(elapsed, 2)
        all_results.append(result)

        print(f"\nAccuracy: {result['correct']}/{result['total']} ({result['accuracy']*100:.1f}%)")
        print(f"Time: {elapsed:.1f}s")

    return all_results


def print_summary(results: List[Dict]) -> None:
    """Print a summary comparison table."""
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Technique':<20} {'Correct':<10} {'Accuracy':<12} {'Time':<10}")
    print("-"*52)

    for r in results:
        acc_str = f"{r['accuracy']*100:.1f}%"
        time_str = f"{r['elapsed_seconds']:.1f}s"
        print(f"{r['technique']:<20} {r['correct']}/{r['total']:<7} {acc_str:<12} {time_str:<10}")

    print("-"*52)

    # Find best technique
    best = max(results, key=lambda x: (x['accuracy'], -x['elapsed_seconds']))
    print(f"\nBest technique: {best['technique']} ({best['accuracy']*100:.1f}% accuracy)")


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare prompting techniques on math/reasoning problems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--technique",
        choices=list(TECHNIQUES.keys()) + ["all"],
        default="all",
        help="Which technique to run (or 'all' to compare)"
    )
    p.add_argument("--host", default=DEFAULT_HOST, help="Ollama host URL")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for generation")
    p.add_argument("--samples", type=int, default=SELF_CONSISTENCY_SAMPLES, help="Number of samples for self-consistency")
    p.add_argument("--verbose", action="store_true", help="Show full model responses")
    p.add_argument("--show-prompts", action="store_true", help="Print each prompt before sending to LLM")
    p.add_argument("--list-models", action="store_true", help="List available Ollama models and exit")
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    # Handle --list-models flag
    if args.list_models:
        list_available_models(args.host)
        return

    print("Prompting Techniques Comparison Demo")
    print("="*40)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Problems: {len(PROBLEMS)}")

    # Check Ollama
    assert_ollama_up(args.host)

    # Determine which techniques to run
    if args.technique == "all":
        techniques = list(TECHNIQUES.keys())
    else:
        techniques = [args.technique]

    # Run comparison
    results = run_comparison(
        host=args.host,
        model=args.model,
        techniques=techniques,
        problems=PROBLEMS,
        temperature=args.temperature,
        num_samples=args.samples,
        verbose=args.verbose,
        show_prompts=args.show_prompts
    )

    # Print summary if comparing multiple techniques
    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main(sys.argv[1:])
