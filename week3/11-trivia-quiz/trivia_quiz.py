#!/usr/bin/env python3
"""
trivia_quiz.py — Interactive multi-turn trivia game powered by Ollama

WHAT THIS IS
------------
A single-file interactive trivia quiz that uses Ollama's /api/chat endpoint
to generate multiple-choice questions and judge answers. The entire quiz runs
as a multi-turn conversation, so Ollama has context of prior questions and
naturally avoids repeats.

WHY THIS EXISTS
---------------
Demonstrates three patterns not covered by other Week 3 demos:
  1) Multi-turn /api/chat — maintains a messages[] array across the whole quiz
  2) System prompts at runtime — sets up a "Quiz Master" persona via the system role
  3) JSON mode for structured back-and-forth — both question generation and answer
     judging use format: "json" to get parseable responses in an interactive loop

REQUIREMENTS
------------
- Python 3.9+
- `pip install requests`
- Ollama running locally: `ollama serve`
- A model pulled: `ollama pull llama3.2`

HOW TO RUN
----------
# Interactive topic selection
python trivia_quiz.py

# Pick topic and difficulty via flags
python trivia_quiz.py --topic Science --questions 3 --difficulty easy

# Use a different model
python trivia_quiz.py --model llama3.1 --topic History

# List available Ollama models
python trivia_quiz.py --list-models
"""

import argparse
import json
import sys
from dataclasses import dataclass
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


# -----------------------------
# Constants
# -----------------------------
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_QUESTIONS = 5

DEFAULT_TOPIC = "Pop Culture"

TOPICS = [
    "Pop Culture", "Science", "History", "Geography",
    "Movies", "Music", "Sports", "Technology", "Literature",
]

SYSTEM_PROMPT = """You are Quiz Master, an enthusiastic and knowledgeable trivia host.

You have two jobs during this quiz:

1) GENERATE QUESTIONS — When asked to generate a trivia question, respond with ONLY valid JSON:
{
  "question": "The question text",
  "options": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
  "correct_answer": "B"
}
Rules for questions:
- Make exactly 4 options (A, B, C, D)
- Exactly one option must be correct
- correct_answer must be one of A, B, C, D
- Do NOT repeat any question you already asked in this conversation
- Make wrong options plausible but clearly wrong to someone who knows the answer

2) JUDGE ANSWERS — When told the user's answer, respond with ONLY valid JSON:
{
  "is_correct": true,
  "explanation": "Brief explanation of the correct answer",
  "fun_fact": "An interesting related fact"
}
Rules for judging:
- is_correct must be a boolean
- explanation should be 1-2 sentences
- fun_fact should be a genuinely interesting tidbit related to the question

Always respond with ONLY the JSON object. No extra text."""


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class QuizQuestion:
    question: str
    options: Dict[str, str]
    correct_answer: str


@dataclass
class Judgment:
    is_correct: bool
    explanation: str
    fun_fact: str


# -----------------------------
# Ollama interaction
# -----------------------------
def chat(host: str, model: str, messages: List[dict], use_json: bool = False) -> str:
    """Send a chat request to Ollama and return the assistant's response content."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if use_json:
        payload["format"] = "json"

    response = requests.post(
        f"{host}/api/chat",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    result = response.json()
    return result["message"]["content"]


def generate_question(
    host: str, model: str, messages: List[dict], topic: str, difficulty: str
) -> QuizQuestion:
    """Ask Ollama to generate a trivia question. Appends messages in-place."""
    difficulty_note = "" if difficulty == "mixed" else f" Difficulty: {difficulty}."
    user_msg = {
        "role": "user",
        "content": (
            f"Generate a {topic} trivia question with 4 multiple-choice options."
            f"{difficulty_note} Respond with JSON only."
        ),
    }
    messages.append(user_msg)

    content = chat(host, model, messages, use_json=True)
    messages.append({"role": "assistant", "content": content})

    data = json.loads(content)
    return QuizQuestion(
        question=data["question"],
        options=data["options"],
        correct_answer=data["correct_answer"].upper(),
    )


def judge_answer(
    host: str, model: str, messages: List[dict],
    question: QuizQuestion, user_answer: str,
) -> Judgment:
    """Ask Ollama to judge whether the user's answer is correct. Appends messages in-place."""
    user_msg = {
        "role": "user",
        "content": (
            f"The user answered: {user_answer} ({question.options.get(user_answer, '')}).\n"
            f"The correct answer is: {question.correct_answer} "
            f"({question.options.get(question.correct_answer, '')}).\n"
            f"Judge whether the user is correct and provide an explanation and fun fact. "
            f"Respond with JSON only."
        ),
    }
    messages.append(user_msg)

    content = chat(host, model, messages, use_json=True)
    messages.append({"role": "assistant", "content": content})

    data = json.loads(content)
    return Judgment(
        is_correct=bool(data.get("is_correct", False)),
        explanation=data.get("explanation", ""),
        fun_fact=data.get("fun_fact", ""),
    )


# -----------------------------
# Display helpers
# -----------------------------
def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")


def display_question(q: QuizQuestion, num: int, total: int):
    """Print a formatted trivia question."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}Question {num}/{total}{Colors.RESET}")
    print(f"{Colors.BOLD}{q.question}{Colors.RESET}\n")
    option_colors = {
        "A": Colors.CYAN,
        "B": Colors.GREEN,
        "C": Colors.YELLOW,
        "D": Colors.HEADER,
    }
    for letter in ("A", "B", "C", "D"):
        if letter in q.options:
            color = option_colors.get(letter, "")
            print(f"  {color}{Colors.BOLD}{letter}){Colors.RESET} {q.options[letter]}")
    print()


def display_result(judgment: Judgment, correct_answer: str):
    """Print whether the user was correct, plus explanation and fun fact."""
    if judgment.is_correct:
        print(f"{Colors.GREEN}{Colors.BOLD}  Correct!{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}  Incorrect!{Colors.RESET} "
              f"The answer was {Colors.BOLD}{correct_answer}{Colors.RESET}")
    print(f"  {Colors.DIM}{judgment.explanation}{Colors.RESET}")
    if judgment.fun_fact:
        print(f"  {Colors.CYAN}Fun fact:{Colors.RESET} {judgment.fun_fact}")


def display_score(correct: int, total: int):
    """Show the running score."""
    print(f"\n  {Colors.BOLD}Score: {correct}/{total}{Colors.RESET}")
    print(f"  {Colors.DIM}{'─' * 40}{Colors.RESET}")


def display_final_score(correct: int, total: int):
    """Show the final scoreboard with a performance tier."""
    pct = (correct / total * 100) if total > 0 else 0
    print_header("Final Score")
    print(f"\n  {Colors.BOLD}{correct} / {total} correct ({pct:.0f}%){Colors.RESET}\n")

    if pct == 100:
        tier = f"{Colors.GREEN}PERFECT SCORE! You're a trivia legend!"
    elif pct >= 80:
        tier = f"{Colors.GREEN}Excellent! Very impressive knowledge!"
    elif pct >= 60:
        tier = f"{Colors.YELLOW}Good job! Solid performance!"
    elif pct >= 40:
        tier = f"{Colors.YELLOW}Not bad! Room to improve."
    else:
        tier = f"{Colors.RED}Better luck next time!"
    print(f"  {Colors.BOLD}{tier}{Colors.RESET}\n")


# -----------------------------
# User input
# -----------------------------
def get_user_answer() -> str:
    """Prompt the user for an answer, validating A/B/C/D."""
    valid = {"A", "B", "C", "D"}
    while True:
        try:
            answer = input(f"{Colors.BOLD}Your answer (A/B/C/D): {Colors.RESET}").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if answer in valid:
            return answer
        print(f"  {Colors.RED}Please enter A, B, C, or D.{Colors.RESET}")


def select_topic() -> str:
    """Show a numbered menu and let the user pick a topic."""
    print(f"\n{Colors.BOLD}Choose a topic:{Colors.RESET}\n")
    for i, topic in enumerate(TOPICS, 1):
        print(f"  {Colors.CYAN}{i}){Colors.RESET} {topic}")
    print(f"  {Colors.CYAN}{len(TOPICS) + 1}){Colors.RESET} Custom topic")

    while True:
        try:
            choice = input(f"\n{Colors.BOLD}Enter number: {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(TOPICS):
                return TOPICS[idx - 1]
            if idx == len(TOPICS) + 1:
                try:
                    custom = input(f"{Colors.BOLD}Enter your topic: {Colors.RESET}").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    sys.exit(0)
                if custom:
                    return custom
        print(f"  {Colors.RED}Please enter a number 1-{len(TOPICS) + 1}.{Colors.RESET}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive trivia quiz powered by Ollama",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use")
    p.add_argument("--host", default=DEFAULT_HOST, help="Ollama host URL")
    p.add_argument("--topic", default=DEFAULT_TOPIC, help="Quiz topic")
    p.add_argument("--questions", type=int, default=DEFAULT_QUESTIONS, help="Number of questions")
    p.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "mixed"],
        default="mixed",
        help="Question difficulty",
    )
    p.add_argument("--list-models", action="store_true", help="List available Ollama models and exit")
    return p.parse_args()


def list_models(host: str):
    """Print available Ollama models and exit."""
    try:
        resp = requests.get(f"{host}/api/tags")
        resp.raise_for_status()
        models = resp.json().get("models", [])
        if not models:
            print("No models found. Pull one with: ollama pull llama3.2")
            return
        print(f"\n{Colors.BOLD}Available models:{Colors.RESET}")
        for m in models:
            name = m.get("name", "unknown")
            size = m.get("size", 0) / (1024 ** 3)
            print(f"  {Colors.CYAN}{name}{Colors.RESET} ({size:.1f} GB)")
    except RequestException as e:
        print(f"{Colors.RED}Could not connect to Ollama at {host}: {e}{Colors.RESET}")


def check_ollama(host: str) -> bool:
    """Verify Ollama is reachable."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        return resp.status_code == 200
    except RequestException:
        return False


# -----------------------------
# Main game loop
# -----------------------------
def run_quiz(host: str, model: str, topic: str, num_questions: int, difficulty: str):
    """Run one round of the trivia quiz."""
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    correct_count = 0

    print_header(f"Trivia Quiz: {topic}")
    print(f"  {Colors.DIM}Model: {model} | Questions: {num_questions} | "
          f"Difficulty: {difficulty}{Colors.RESET}")

    for i in range(1, num_questions + 1):
        # Generate question
        try:
            question = generate_question(host, model, messages, topic, difficulty)
        except (RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"\n{Colors.RED}Error generating question: {e}{Colors.RESET}")
            print("Skipping this question...\n")
            continue

        display_question(question, i, num_questions)
        user_answer = get_user_answer()

        # Judge answer
        try:
            judgment = judge_answer(host, model, messages, question, user_answer)
        except (RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"\n{Colors.RED}Error judging answer: {e}{Colors.RESET}")
            # Fall back to simple string comparison
            is_correct = user_answer == question.correct_answer
            judgment = Judgment(
                is_correct=is_correct,
                explanation=f"The correct answer was {question.correct_answer}.",
                fun_fact="",
            )

        if judgment.is_correct:
            correct_count += 1

        display_result(judgment, question.correct_answer)
        display_score(correct_count, i)

    display_final_score(correct_count, num_questions)
    return correct_count


def main():
    args = parse_args()

    if args.list_models:
        list_models(args.host)
        sys.exit(0)

    # Check Ollama connectivity
    if not check_ollama(args.host):
        print(f"{Colors.RED}Cannot connect to Ollama at {args.host}{Colors.RESET}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    print_header("Ollama Trivia Quiz")
    print(f"  {Colors.DIM}Powered by {args.model} via Ollama{Colors.RESET}")

    topic = args.topic

    while True:
        run_quiz(args.host, args.model, topic, args.questions, args.difficulty)

        # Play again?
        try:
            again = input(f"\n{Colors.BOLD}Play again? (y/n): {Colors.RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if again != "y":
            break

        # Let user pick a new topic
        try:
            new_topic = input(
                f"{Colors.BOLD}Same topic ({topic})? Press Enter or type a new one: {Colors.RESET}"
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if new_topic:
            topic = new_topic

    print(f"\n{Colors.CYAN}Thanks for playing!{Colors.RESET}\n")


if __name__ == "__main__":
    main()
