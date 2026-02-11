# Prompting Techniques Comparison

A single-file demo comparing how different prompting strategies affect LLM accuracy on math and reasoning problems.

## Why This Matters

Prompting technique can dramatically improve accuracy on reasoning tasks—often more than switching models. This demo lets you see the effect firsthand with challenging problems designed to differentiate techniques.

## Available Techniques

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| `zero-shot` | Just ask the question directly | Baseline; simple factual queries |
| `few-shot` | Provide worked examples first | When format/style matters |
| `cot` | Chain-of-thought: ask model to reason step-by-step | Multi-step reasoning problems |
| `few-shot-cot` | Examples with reasoning + step-by-step | Complex problems needing both |
| `self-consistency` | Sample multiple CoT responses, majority vote | High-stakes decisions |
| `role` | Add expert persona ("You are an expert...") | Domain-specific problems |
| `step-back` | First identify problem type, then solve | Abstract reasoning |
| `least-to-most` | Break into subproblems, solve sequentially | Complex multi-part problems |

## Quick Start

```bash
# Run all techniques and compare results
python prompting_techniques.py

# Run a specific technique
python prompting_techniques.py --technique cot

# See what prompts are being sent to the LLM
python prompting_techniques.py --technique cot --show-prompts

# List available Ollama models
python prompting_techniques.py --list-models

# Use a different model
python prompting_techniques.py --model llama3.1

# Adjust self-consistency samples
python prompting_techniques.py --technique self-consistency --samples 7
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--technique` | `all` | Which technique to run (or `all` to compare) |
| `--model` | `llama3.2` | Ollama model to use |
| `--temperature` | `0.0` | Temperature for generation (higher = more random) |
| `--samples` | `5` | Number of samples for self-consistency |
| `--verbose` | off | Show full model responses |
| `--show-prompts` | off | Print each prompt before sending to LLM |
| `--list-models` | off | List available Ollama models and exit |
| `--host` | `http://localhost:11434` | Ollama host URL |

## Test Problems

The test set includes 8 challenging problems designed to trip up LLMs without proper reasoning:

1. **Distractor** - Includes irrelevant numbers that confuse the model
2. **Backwards reasoning** - Must work from result back to original value
3. **Multi-step with state** - Track values through sequential operations
4. **Careful reading** - Must extract all relationships correctly
5. **Rate change trap** - Average speed isn't the average of two speeds
6. **Language trap** - "All but 9 run away" phrasing
7. **Nested percentages** - Sequential discounts compound
8. **Work rate** - Requires understanding combined work rates

## Example Results

Typical results on llama3.2 (your mileage may vary):

| Technique | Score |
|-----------|-------|
| zero-shot | 0/8 (0%) |
| few-shot | 4/8 (50%) |
| step-back | 5/8 (62.5%) |
| cot | 6/8 (75%) |
| few-shot-cot | 6/8 (75%) |

## Understanding the Output

When you run with `--show-prompts`, you'll see each prompt wrapped in visual separators:

```
────────────────────────────────────────────────────────────
PROMPT (cot):
────────────────────────────────────────────────────────────
A tank is being filled with water...

Let's think step by step to solve this problem...
────────────────────────────────────────────────────────────
```

Results show expected vs predicted answers:
```
  multi_step_1: expected=175, got=175 [/] correct
  sequence_trap: expected=9, got=8 [X] wrong
```

## Exercises

1. **Compare models**: Run `--technique cot` with different models (`--model llama3.1`, `--model gemma3:4b`) and compare accuracy
2. **Study prompts**: Use `--show-prompts` to see how each technique structures its prompt differently
3. **Temperature effects**: Try `--temperature 0.7` vs `--temperature 0.0` and observe consistency
4. **Self-consistency tuning**: Experiment with `--samples 3` vs `--samples 10` for self-consistency

## Requirements

- Python 3.9+
- `pip install requests`
- Ollama running locally: `ollama serve`
- A model pulled: `ollama pull llama3.2`
