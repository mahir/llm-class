from openai import OpenAI
import json
import jsonschema
import time

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
                "enum": ["calm", "frustrated", "angry", "unclear"],
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
                ],
            },
        },
        "required": ["intent", "priority", "customer_tone", "summary", "next_action"],
        "additionalProperties": False,
    },
}

email_text = """
Subject: My order 137-884 arrived broken
Message: The mug is cracked. I need a refund. This is the second time this happens.
"""


def triage_with_retry(email_text, max_retries=2):
    error_msg = ""
    for attempt in range(max_retries + 1):
        prompt = f"""
Normalize this customer email into the required schema. If data is missing, use null.
{("Important: " + error_msg) if error_msg else ""}
Text:
{email_text}
"""
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": TICKET_SCHEMA,
            },
        )

        raw = resp.choices[0].message.content

        try:
            data = json.loads(raw)
            jsonschema.validate(instance=data, schema=TICKET_SCHEMA["schema"])
            return data
        except (json.JSONDecodeError, jsonschema.ValidationError):
            error_msg = (
                "Your output must be a single JSON object that passes the provided schema. No extra text."
            )
            if attempt == max_retries:
                raise
            time.sleep(0.5)


ticket = triage_with_retry(email_text)
print(ticket)

