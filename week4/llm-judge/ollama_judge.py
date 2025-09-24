#!/usr/bin/env python3
"""
Ollama Model Comparison Judge

A script that runs two LLM models side by side using Ollama,
then uses a third model to evaluate and compare their responses.
"""

import json
import time
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    exit(1)


class JudgmentCriteria(Enum):
    """Evaluation criteria for comparing model responses."""
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    HELPFULNESS = "helpfulness"


@dataclass
class ModelResponse:
    """Container for a model's response and metadata."""
    model: str
    response: str
    response_time: float
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Container for the comparison evaluation."""
    winner: str
    reasoning: str
    scores: Dict[str, int]  # Scores out of 10 for each model


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        except requests.RequestException as e:
            print(f"Error fetching models: {e}")
            return []
    
    def generate_response(self, model: str, prompt: str, 
                         temperature: float = 0.7) -> ModelResponse:
        """Generate response from a specific model."""
        start_time = time.time()
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            response_time = time.time() - start_time
            
            return ModelResponse(
                model=model,
                response=result.get('response', ''),
                response_time=response_time
            )
            
        except requests.RequestException as e:
            response_time = time.time() - start_time
            return ModelResponse(
                model=model,
                response="",
                response_time=response_time,
                error=str(e)
            )


class ModelJudge:
    """Judges and compares responses from different models."""
    
    def __init__(self, ollama_client: OllamaClient, 
                 judge_model: str = "qwen3:32b"):
        self.client = ollama_client
        self.judge_model = judge_model
    
    def _create_judgment_prompt(self, original_prompt: str, 
                               response_a: ModelResponse, 
                               response_b: ModelResponse) -> str:
        """Create prompt for the judge model."""
        return f"""You are an expert evaluator. Compare these two AI responses to the same question.

ORIGINAL QUESTION:
{original_prompt}

RESPONSE A (from {response_a.model}):
{response_a.response}

RESPONSE B (from {response_b.model}):
{response_b.response}

Please evaluate both responses and provide your analysis in this exact format:

WINNER: [A or B or Tie]
REASONING: [Explain in 2-3 sentences why one response is better, considering accuracy, clarity, completeness, and helpfulness. Be specific about what makes the winning response superior.]
SCORES: A=X, B=Y

Replace X and Y with scores from 1-10. Be decisive in your evaluation."""
    
    def evaluate_responses(self, original_prompt: str, 
                          response_a: ModelResponse, 
                          response_b: ModelResponse) -> ComparisonResult:
        """Evaluate and compare two model responses."""
        judgment_prompt = self._create_judgment_prompt(
            original_prompt, response_a, response_b
        )
        
        judgment = self.client.generate_response(
            self.judge_model, judgment_prompt, temperature=0.3
        )
        
        if judgment.error:
            return ComparisonResult(
                winner="Error",
                reasoning=f"Judge model error: {judgment.error}",
                scores={response_a.model: 0, response_b.model: 0}
            )
        
        # Parse judgment response
        return self._parse_judgment(judgment.response, response_a, response_b)
    
    def _parse_judgment(self, judgment_text: str, 
                       response_a: ModelResponse, 
                       response_b: ModelResponse) -> ComparisonResult:
        """Parse the judge model's response with robust parsing."""
        winner = "Tie"
        reasoning = ""
        scores = {response_a.model: 5, response_b.model: 5}
        
        # Debug: Print raw judgment for troubleshooting
        print(f"\n🔍 DEBUG - Raw judge response:")
        print(f"'{judgment_text}'")
        print("-" * 40)
        
        # Try structured parsing first
        lines = judgment_text.strip().split('\n')
        reasoning_lines = []
        collecting_reasoning = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for winner
            if line.upper().startswith("WINNER:"):
                winner_letter = line.split(":", 1)[1].strip().upper()
                if winner_letter == "A":
                    winner = response_a.model
                elif winner_letter == "B":
                    winner = response_b.model
                else:
                    winner = "Tie"
            
            # Look for reasoning
            elif line.upper().startswith("REASONING:"):
                reasoning_content = line.split(":", 1)[1].strip()
                if reasoning_content:
                    reasoning_lines.append(reasoning_content)
                collecting_reasoning = True
            
            # Continue collecting reasoning if we started
            elif collecting_reasoning and not line.upper().startswith("SCORES:"):
                reasoning_lines.append(line)
            
            # Look for scores
            elif line.upper().startswith("SCORES:"):
                collecting_reasoning = False
                try:
                    scores_text = line.split(":", 1)[1].strip()
                    # Parse various formats: "A_score=X, B_score=Y" or "A=X, B=Y" etc.
                    import re
                    a_match = re.search(r'A[_\s]*(?:score)?[_\s]*[=:]\s*(\d+)', scores_text, re.IGNORECASE)
                    b_match = re.search(r'B[_\s]*(?:score)?[_\s]*[=:]\s*(\d+)', scores_text, re.IGNORECASE)
                    
                    if a_match:
                        scores[response_a.model] = int(a_match.group(1))
                    if b_match:
                        scores[response_b.model] = int(b_match.group(1))
                except (ValueError, IndexError):
                    pass  # Keep default scores
        
        # Join reasoning lines
        if reasoning_lines:
            reasoning = " ".join(reasoning_lines).strip()
        
        # Fallback: If structured parsing failed, use the whole response as reasoning
        if not reasoning:
            reasoning = judgment_text.strip()
            
            # Try to extract winner from free-form text
            text_lower = judgment_text.lower()
            if "response a" in text_lower and ("better" in text_lower or "winner" in text_lower):
                if text_lower.find("response a") < text_lower.find("response b"):
                    winner = response_a.model
            elif "response b" in text_lower and ("better" in text_lower or "winner" in text_lower):
                winner = response_b.model
        
        return ComparisonResult(winner=winner, reasoning=reasoning, scores=scores)


class ModelComparator:
    """Main class orchestrating model comparisons."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.client = OllamaClient(ollama_url)
        self.judge = None
    
    def setup_judge(self, judge_model: str = "qwen3:32b") -> bool:
        """Setup the judge model."""
        available_models = self.client.list_models()
        if judge_model not in available_models:
            print(f"Judge model '{judge_model}' not available.")
            print(f"Available models: {', '.join(available_models)}")
            return False
        
        self.judge = ModelJudge(self.client, judge_model)
        return True
    
    def compare_models(self, model_a: str, model_b: str, prompt: str,
                      temperature: float = 0.7) -> Dict:
        """Compare two models on the same prompt."""
        if not self.client.is_available():
            raise ConnectionError("Ollama server is not available")
        
        if not self.judge:
            raise ValueError("Judge model not set up")
        
        print(f"Generating response from {model_a}...")
        response_a = self.client.generate_response(model_a, prompt, temperature)
        
        print(f"Generating response from {model_b}...")
        response_b = self.client.generate_response(model_b, prompt, temperature)
        
        if response_a.error or response_b.error:
            return {
                "error": "One or both models failed to respond",
                "model_a_error": response_a.error,
                "model_b_error": response_b.error
            }
        
        print("Evaluating responses...")
        comparison = self.judge.evaluate_responses(prompt, response_a, response_b)
        
        return {
            "original_prompt": prompt,
            "model_a": {
                "name": model_a,
                "response": response_a.response,
                "response_time": response_a.response_time
            },
            "model_b": {
                "name": model_b,
                "response": response_b.response,
                "response_time": response_b.response_time
            },
            "evaluation": {
                "winner": comparison.winner,
                "reasoning": comparison.reasoning,
                "scores": comparison.scores
            }
        }
    
    def print_comparison_results(self, results: Dict):
        """Pretty print comparison results."""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        print(f"\n📝 Original Prompt:")
        print(f"{results['original_prompt']}")
        
        print(f"\n🤖 {results['model_a']['name']} Response:")
        print(f"⏱️  Response time: {results['model_a']['response_time']:.2f}s")
        print(f"{'-'*40}")
        print(f"{results['model_a']['response']}")
        
        print(f"\n🤖 {results['model_b']['name']} Response:")
        print(f"⏱️  Response time: {results['model_b']['response_time']:.2f}s")
        print(f"{'-'*40}")
        print(f"{results['model_b']['response']}")
        
        print(f"\n⚖️  Evaluation:")
        print(f"🏆 Winner: {results['evaluation']['winner']}")
        print(f"📊 Scores: {results['evaluation']['scores']}")
        print(f"💭 Reasoning:")
        # Word wrap reasoning for better readability
        reasoning = results['evaluation']['reasoning']
        if reasoning:
            import textwrap
            wrapped_reasoning = textwrap.fill(reasoning, width=70, initial_indent="   ", subsequent_indent="   ")
            print(wrapped_reasoning)
        else:
            print("   [No reasoning provided]")
        print("="*80)


def main():
    """Example usage and CLI interface."""
    parser = argparse.ArgumentParser(description="Compare two Ollama models")
    parser.add_argument("--model-a", default="gemma3:1b", 
                       help="First model to compare")
    parser.add_argument("--model-b", default="qwen3:4b", 
                       help="Second model to compare")
    parser.add_argument("--judge", default="qwen3:32b", 
                       help="Model to use as judge")
    parser.add_argument("--prompt", 
                       help="Prompt to test both models with")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for model responses")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                       help="Ollama server URL")
    
    args = parser.parse_args()
    
    # Example prompts if none provided
    if not args.prompt:
        example_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate the Fibonacci sequence.",
            "What are the main causes of climate change?",
            "How do you make a perfect cup of coffee?",
            "Explain the difference between AI and machine learning."
        ]
        
        print("No prompt provided. Choose from these examples:")
        for i, prompt in enumerate(example_prompts, 1):
            print(f"{i}. {prompt}")
        
        choice = input("\nEnter choice (1-5) or type your own prompt: ")
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(example_prompts):
                args.prompt = example_prompts[choice_idx]
        except ValueError:
            args.prompt = choice
    
    # Initialize comparator
    comparator = ModelComparator(args.ollama_url)
    
    # Check if Ollama is available
    if not comparator.client.is_available():
        print("❌ Ollama server is not running. Please start Ollama first.")
        return 1
    
    # List available models
    available_models = comparator.client.list_models()
    print(f"Available models: {', '.join(available_models)}")
    
    # Verify models exist
    if args.model_a not in available_models:
        print(f"❌ Model '{args.model_a}' not found. Please pull it first:")
        print(f"   ollama pull {args.model_a}")
        return 1
    
    if args.model_b not in available_models:
        print(f"❌ Model '{args.model_b}' not found. Please pull it first:")
        print(f"   ollama pull {args.model_b}")
        return 1
    
    # Setup judge
    if not comparator.setup_judge(args.judge):
        return 1
    
    try:
        # Run comparison
        results = comparator.compare_models(
            args.model_a, args.model_b, args.prompt, args.temperature
        )
        
        # Display results
        comparator.print_comparison_results(results)
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())