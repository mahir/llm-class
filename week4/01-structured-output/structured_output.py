#!/usr/bin/env python3
"""
Ollama Structured Output Examples
Shows how to get structured JSON responses from Ollama, progressing from
simple extraction to complex multi-layered business analysis.
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
        "model": "llama3.2",  # Change this to your preferred model
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


# ---------------------------------------------------------------------------
# Example 1 (Simple): Extract person information
# ---------------------------------------------------------------------------
def example_1_person_info():
    """Extract structured person data from free text"""
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


# ---------------------------------------------------------------------------
# Example 2 (Medium): Product review sentiment analysis
# ---------------------------------------------------------------------------
def example_2_product_review():
    """Analyze a product review into structured sentiment data"""
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


# ---------------------------------------------------------------------------
# Example 3 (Medium): Task breakdown
# ---------------------------------------------------------------------------
def example_3_task_breakdown():
    """Break down a task into structured subtasks"""
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


# ---------------------------------------------------------------------------
# Example 4 (Complex): Business scenario analysis
# ---------------------------------------------------------------------------
def example_4_business_analysis():
    """Complex business scenario analysis with deeply nested JSON schema"""
    print("=== Example 4: Complex Business Scenario Analysis ===")

    prompt = """
    Analyze this comprehensive business scenario for TechFlow Solutions, a mid-sized software consulting company:

    COMPANY BACKGROUND:
    TechFlow Solutions has been operating for 8 years with 150 employees across 3 offices (San Francisco HQ with 80 employees, Austin satellite with 45 employees, and Denver remote hub with 25 employees). The company specializes in enterprise software development, cloud migration services, and AI/ML consulting. Annual revenue is $18.5M with 22% profit margins. Current client base includes 47 active clients, with top 5 clients representing 60% of revenue.

    CURRENT SITUATION:
    The company is facing multiple challenges simultaneously. First, their largest client (GlobalTech Corp, representing 28% of annual revenue) is threatening to terminate their $5.2M annual contract due to delivery delays on a critical AI recommendation engine project. The delays stem from technical complexity, resource constraints, and integration issues with the client's legacy systems. The project is 4 months behind schedule and 40% over budget.

    Second, there's internal turmoil with high employee turnover (18% last quarter vs 8% industry average). Exit interviews reveal concerns about work-life balance, limited career progression, outdated technology stack, and compensation gaps (15-20% below market rate for senior developers). The Austin office is particularly affected with 3 senior architects leaving in 6 weeks.

    Third, the competitive landscape has intensified. Two major competitors (CloudScale Dynamics and NextGen Solutions) have opened offices in their primary markets, offering similar services at 15-20% lower rates while promising faster delivery times. They're also poaching key talent with 25-30% salary increases and equity packages.

    FINANCIAL PRESSURES:
    Cash flow has tightened due to delayed payments from 3 major clients totaling $2.8M (30-90 days overdue). The company has $1.2M in operating expenses monthly and only 2.5 months of runway at current burn rate. Bank credit line is at 80% utilization ($800K of $1M limit). Recent equipment purchases for the AI/ML division ($450K in GPU infrastructure) have strained resources.

    OPPORTUNITIES:
    Despite challenges, there are growth opportunities. The AI/ML consulting division has a $3.2M pipeline with 12 qualified prospects, including 2 Fortune 500 companies. A potential acquisition opportunity exists with DataViz Pro, a 25-person data visualization company valued at $4.5M, which would add complementary capabilities and $2.8M annual revenue. Additionally, a strategic partnership opportunity with CloudMega (major cloud provider) could provide preferred vendor status and co-marketing opportunities.

    STRATEGIC DECISIONS NEEDED:
    CEO Sarah Mitchell must make critical decisions in the next 30 days: 1) Whether to invest additional resources ($800K) to save the GlobalTech contract vs cutting losses, 2) How to address the talent retention crisis while managing costs, 3) Whether to pursue the DataViz Pro acquisition despite cash constraints, 4) How to respond to competitive pricing pressure without destroying margins, 5) Whether to consolidate offices to reduce overhead ($180K annually) vs maintaining distributed presence for talent access.

    STAKEHOLDER PERSPECTIVES:
    Board members are split - two favor aggressive cost-cutting and office consolidation, while one pushes for growth investments and acquisition. The executive team has mixed views: CTO favors technology investments and talent retention, CFO prioritizes cash preservation and cost reduction, VP of Sales wants to match competitor pricing to protect market share. Employees are anxious about layoffs and office closures, while clients are questioning the company's stability and ability to deliver.

    Analyze this scenario and provide comprehensive strategic recommendations with risk assessments, financial implications, timeline considerations, and implementation priorities.
    """

    schema = """{
        "scenario_summary": {
            "company_name": "string",
            "industry": "string",
            "company_size": "string",
            "annual_revenue": "string",
            "primary_challenges": ["array of strings"],
            "key_opportunities": ["array of strings"]
        },
        "financial_analysis": {
            "current_cash_position": "string",
            "monthly_burn_rate": "string",
            "runway_months": "number",
            "revenue_at_risk": "string",
            "critical_metrics": {
                "profit_margin": "string",
                "client_concentration_risk": "string",
                "employee_turnover_rate": "string"
            }
        },
        "strategic_recommendations": [
            {
                "priority": "high/medium/low",
                "recommendation": "string",
                "rationale": "string",
                "implementation_timeline": "string",
                "required_investment": "string",
                "expected_roi": "string",
                "risk_level": "high/medium/low",
                "success_metrics": ["array of strings"]
            }
        ],
        "risk_assessment": {
            "immediate_risks": [
                {
                    "risk": "string",
                    "probability": "high/medium/low",
                    "impact": "high/medium/low",
                    "mitigation_strategy": "string"
                }
            ],
            "long_term_risks": [
                {
                    "risk": "string",
                    "probability": "high/medium/low",
                    "impact": "high/medium/low",
                    "mitigation_strategy": "string"
                }
            ]
        },
        "implementation_roadmap": {
            "phase_1_immediate": {
                "timeframe": "string",
                "key_actions": ["array of strings"],
                "success_criteria": ["array of strings"]
            },
            "phase_2_short_term": {
                "timeframe": "string",
                "key_actions": ["array of strings"],
                "success_criteria": ["array of strings"]
            },
            "phase_3_long_term": {
                "timeframe": "string",
                "key_actions": ["array of strings"],
                "success_criteria": ["array of strings"]
            }
        }
    }"""

    result = get_structured_response(prompt, schema)
    print(f"Complex Business Scenario (excerpt): {prompt[:200]}...")
    print(f"Full prompt length: {len(prompt)} characters")
    print(f"\nStructured Analysis Output:")
    print(json.dumps(result, indent=2))
    print()


def main():
    """Run all examples, progressing from simple to complex"""
    print("Ollama Structured Output Examples")
    print("=================================")
    print("Make sure Ollama is running with: ollama serve")
    print("And you have a model installed: ollama pull llama3.2")
    print()

    # Run examples in order of complexity
    example_1_person_info()
    example_2_product_review()
    example_3_task_breakdown()
    example_4_business_analysis()

    print("=== Custom Example ===")
    print("Try your own prompt:")

    custom_prompt = input("Enter your prompt (or press Enter to skip): ")
    if custom_prompt:
        custom_schema = input("Enter your JSON schema description: ")
        if custom_schema:
            result = get_structured_response(custom_prompt, custom_schema)
            print(f"Structured Output: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
