from openai import OpenAI
import json
import jsonschema
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

client = OpenAI(api_key="")
# or use env var: export OPENAI_API_KEY="..."

TICKET_SCHEMA = {
    "name": "SupportTicket",
    "schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [
                    "refund_request",
                    "order_status",
                    "account_issue",
                    "product_question",
                    "technical_support",
                    "billing_inquiry",
                    "other"
                ],
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high", "urgent"],
            },
            "order_id": {"type": ["string", "null"]},
            "item_name": {"type": ["string", "null"]},
            "customer_tone": {
                "type": "string",
                "enum": ["calm", "frustrated", "angry", "confused", "polite"],
            },
            "summary": {"type": "string"},
            "next_action": {
                "type": "string",
                "enum": [
                    "approve_refund",
                    "request_more_info",
                    "expedite_shipping",
                    "route_to_agent",
                    "answer_with_faq",
                    "escalate_to_manager",
                    "provide_tracking_info",
                ],
            },
            "estimated_resolution_time": {
                "type": "string",
                "enum": ["immediate", "24_hours", "3_days", "1_week"]
            }
        },
        "required": ["intent", "priority", "customer_tone", "summary", "next_action", "estimated_resolution_time"],
        "additionalProperties": False,
    },
}

# Multiple test examples
TEST_EMAILS = [
    {
        "name": "Broken Item Refund",
        "email": """
Subject: My order 137-884 arrived broken
Message: The mug is cracked. I need a refund. This is the second time this happens.
""",
        "expected_intent": "refund_request"
    },
    {
        "name": "Order Status Inquiry",
        "email": """
Subject: Where is my package?
Message: Hi, I ordered a laptop case last week (order #245-991) but haven't received any tracking information. 
Can you please let me know when it will arrive? I need it for a business trip next week.
""",
        "expected_intent": "order_status"
    },
    {
        "name": "Account Login Issues",
        "email": """
Subject: Can't access my account!!
Message: I've been trying to log in for 2 hours and it keeps saying my password is wrong. 
I tried resetting it 3 times but never got the email. This is really frustrating - I need to check my orders!
""",
        "expected_intent": "account_issue"
    },
    {
        "name": "Product Compatibility Question",
        "email": """
Subject: Question about phone case compatibility
Message: Hello, I'm interested in the clear phone case but want to make sure it fits the iPhone 15 Pro Max. 
The product page doesn't specifically mention this model. Could you confirm compatibility? Thanks!
""",
        "expected_intent": "product_question"
    },
    {
        "name": "Billing Dispute",
        "email": """
Subject: Charged twice for the same order
Message: I just checked my credit card statement and I was charged $89.99 twice for order #556-123. 
I only placed one order. Please resolve this immediately as it's affecting my available credit.
""",
        "expected_intent": "billing_inquiry"
    },
    {
        "name": "Technical Support",
        "email": """
Subject: Software not working after update
Message: After the latest app update, the barcode scanner feature stopped working on my Android phone. 
I've tried restarting the app and my phone but nothing works. I need this for my job.
""",
        "expected_intent": "technical_support"
    },
    {
        "name": "Urgent Shipping Request",
        "email": """
Subject: URGENT: Need order expedited
Message: I placed order #789-456 yesterday for my daughter's birthday gift which is TOMORROW. 
I'm willing to pay extra for overnight shipping. Please help!!!
""",
        "expected_intent": "order_status"
    }
]


@dataclass
class TriageResult:
    """Class to hold triage results with metadata"""
    ticket_data: Dict
    success: bool
    attempts_used: int
    error_message: Optional[str] = None


class TicketTriager:
    """Enhanced ticket triage system with better error handling and logging"""
    
    def __init__(self, client: OpenAI, schema: Dict, model: str = "gpt-4o-2024-08-06"):
        self.client = client
        self.schema = schema
        self.model = model
    
    def triage_with_retry(self, email_text: str, max_retries: int = 3) -> TriageResult:
        """
        Triage a customer email with retry logic and detailed error tracking
        """
        error_msg = ""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                prompt = self._build_prompt(email_text, error_msg)
                
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": self.schema,
                    },
                    temperature=0.1  # Lower temperature for more consistent results
                )

                raw_response = resp.choices[0].message.content
                
                # Parse and validate the response
                data = json.loads(raw_response)
                jsonschema.validate(instance=data, schema=self.schema["schema"])
                
                return TriageResult(
                    ticket_data=data,
                    success=True,
                    attempts_used=attempt + 1
                )
                
            except json.JSONDecodeError as e:
                last_exception = e
                error_msg = f"JSON parsing failed: {str(e)}. Ensure your output is valid JSON."
                
            except jsonschema.ValidationError as e:
                last_exception = e
                error_msg = f"Schema validation failed: {str(e)}. Check required fields and enum values."
                
            except Exception as e:
                last_exception = e
                error_msg = f"Unexpected error: {str(e)}"
            
            if attempt < max_retries:
                print(f"Attempt {attempt + 1} failed, retrying in 1 second...")
                time.sleep(1)
        
        # All retries failed
        return TriageResult(
            ticket_data={},
            success=False,
            attempts_used=max_retries + 1,
            error_message=f"Failed after {max_retries + 1} attempts. Last error: {str(last_exception)}"
        )
    
    def _build_prompt(self, email_text: str, error_msg: str = "") -> str:
        """Build the prompt with examples and clear instructions"""
        error_section = ("IMPORTANT: " + error_msg + "\n") if error_msg else ""
        
        base_prompt = f"""
You are a customer support ticket triage system. Analyze the following customer email and extract structured information according to the provided JSON schema.

Guidelines:
- If order_id is mentioned in format like "order #123-456" or "order 123-456", extract just the ID part
- If item_name is not explicitly mentioned, use null
- Assess customer_tone based on language used (calm, frustrated, angry, confused, polite)
- Set priority based on urgency indicators and customer tone
- Choose the most appropriate next_action based on the intent and priority
- Provide a concise summary of the main issue

{error_section}

Customer Email:
{email_text}

Return only a valid JSON object that matches the schema exactly.
"""
        return base_prompt
    
    def batch_process(self, emails: List[Dict]) -> List[Dict]:
        """Process multiple emails and return results with metadata"""
        results = []
        
        for i, email_data in enumerate(emails):
            print(f"\nProcessing email {i+1}/{len(emails)}: {email_data['name']}")
            print("-" * 50)
            
            result = self.triage_with_retry(email_data['email'])
            
            output = {
                'name': email_data['name'],
                'success': result.success,
                'attempts_used': result.attempts_used,
                'expected_intent': email_data.get('expected_intent'),
            }
            
            if result.success:
                output['ticket_data'] = result.ticket_data
                actual_intent = result.ticket_data.get('intent')
                expected_intent = email_data.get('expected_intent')
                output['intent_match'] = actual_intent == expected_intent
                
                print(f"✅ Success (attempts: {result.attempts_used})")
                intent_indicator = '✅' if output['intent_match'] else '❌'
                print(f"Intent: {actual_intent} {intent_indicator}")
                print(f"Priority: {result.ticket_data.get('priority')}")
                print(f"Summary: {result.ticket_data.get('summary')[:100]}...")
                
            else:
                output['error'] = result.error_message
                print(f"❌ Failed: {result.error_message}")
            
            results.append(output)
        
        return results


def print_summary(results: List[Dict]):
    """Print a summary of batch processing results"""
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    intent_matches = sum(1 for r in results if r.get('intent_match', False))
    
    separator = "=" * 60
    print(f"\n{separator}")
    print("BATCH PROCESSING SUMMARY")
    print(separator)
    print(f"Total emails processed: {total}")
    print(f"Successful triages: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"Intent accuracy: {intent_matches}/{total} ({intent_matches/total*100:.1f}%)")
    
    # Show failed cases
    failed = [r for r in results if not r['success']]
    if failed:
        print(f"\nFailed cases:")
        for r in failed:
            print(f"  - {r['name']}: {r['error']}")
    
    # Show intent mismatches
    mismatches = [r for r in results if r.get('intent_match') == False]
    if mismatches:
        print(f"\nIntent mismatches:")
        for r in mismatches:
            expected = r['expected_intent']
            actual = r['ticket_data']['intent']
            print(f"  - {r['name']}: expected '{expected}', got '{actual}'")


def main():
    """Main function to demonstrate the enhanced triage system"""
    print("🎫 Enhanced Customer Support Ticket Triage System")
    print("="*60)
    
    # Initialize the triager
    triager = TicketTriager(client, TICKET_SCHEMA)
    
    # Process all test emails
    results = triager.batch_process(TEST_EMAILS)
    
    # Print summary
    print_summary(results)
    
    # Demonstrate single email processing
    separator = "=" * 60
    print(f"\n{separator}")
    print("SINGLE EMAIL EXAMPLE")
    print(separator)
    
    single_result = triager.triage_with_retry(TEST_EMAILS[0]['email'])
    if single_result.success:
        print("Structured ticket data:")
        print(json.dumps(single_result.ticket_data, indent=2))
    else:
        print(f"Processing failed: {single_result.error_message}")


if __name__ == "__main__":
    main()