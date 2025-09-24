#!/usr/bin/env python3
"""
Simple Ollama Structured Output Example
This example shows how to get structured JSON responses from Ollama
"""

import requests
import json

def get_structured_response(prompt, schema_description):
    """
    Get a structured JSON response from Ollama
    
    Args:
        prompt: The user's question or input
        schema_description: Description of the expected JSON structure
    
    Returns:
        dict: Parsed JSON response
    """
    
    # Construct the system prompt for structured output
    system_prompt = f"""You are a helpful assistant that always responds with valid JSON.
    
    Required JSON structure: {schema_description}
    
    Rules:
    - Always respond with valid JSON only
    - No additional text before or after the JSON
    - Follow the exact schema provided
    """
    
    # Prepare the request payload
    payload = {
        "model": "llama3.1",  # Change this to your preferred model
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "stream": False,
        "format": "json"  # This tells Ollama to return JSON
    }
    
    try:
        # Make request to Ollama
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            # Parse the JSON response from the model
            model_response = result["message"]["content"]
            return json.loads(model_response)
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to Ollama. Make sure it's running on localhost:11434"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON response: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

def example_1_person_info():
    """Example 1: Extract person information"""
    print("=== Example 1: Person Information Extraction ===")
    
    prompt = "Tell me about John Smith, a 32-year-old software engineer from Seattle who loves hiking and photography."
    
    schema = """{
        "name": "string",
        "age": "number", 
        "profession": "string",
        "location": "string",
        "hobbies": ["array of strings"]
    }"""
    
    result = get_structured_response(prompt, schema)
    print(f"Input: {prompt}")
    print(f"Structured Output: {json.dumps(result, indent=2)}")
    print()

def example_2_product_review():
    """Example 2: Product review analysis"""
    print("=== Example 2: Product Review Analysis ===")
    
    prompt = "This laptop is amazing! Great performance, beautiful screen, but the battery life could be better. I'd definitely recommend it. 4 out of 5 stars."
    
    schema = """{
        "sentiment": "positive/negative/neutral",
        "rating": "number (1-5)",
        "pros": ["array of strings"],
        "cons": ["array of strings"],
        "recommendation": "boolean"
    }"""
    
    result = get_structured_response(prompt, schema)
    print(f"Input: {prompt}")
    print(f"Structured Output: {json.dumps(result, indent=2)}")
    print()

def example_3_task_breakdown():
    """Example 3: Task breakdown"""
    print("=== Example 3: Task Breakdown ===")
    
    prompt = "I need to plan a birthday party for 20 people next weekend."
    
    schema = """{
        "main_task": "string",
        "estimated_duration": "string",
        "difficulty": "easy/medium/hard",
        "subtasks": [
            {
                "task": "string",
                "priority": "high/medium/low",
                "estimated_time": "string"
            }
        ],
        "required_resources": ["array of strings"]
    }"""
    
    result = get_structured_response(prompt, schema)
    print(f"Input: {prompt}")
    print(f"Structured Output: {json.dumps(result, indent=2)}")
    print()

def main():
    """Run all examples"""
    print("Ollama Structured Output Examples")
    print("=================================")
    print("Make sure Ollama is running with: ollama serve")
    print("And you have a model installed: ollama pull llama3.1")
    print()
    
    # Run examples
    example_1_person_info()
    example_2_product_review() 
    example_3_task_breakdown()
    
    print("=== Custom Example ===")
    print("Try your own prompt:")
    
    custom_prompt = input("Enter your prompt: ")
    custom_schema = input("Enter your JSON schema description: ")
    
    if custom_prompt and custom_schema:
        result = get_structured_response(custom_prompt, custom_schema)
        print(f"Structured Output: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()