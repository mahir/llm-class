#!/usr/bin/env python3
"""
Simple Ollama Batch Processing Example - Tag Extraction
Save as: tag_extractor_batch.py

This example shows how to batch process similar tasks like extracting
tags from product descriptions, analyzing reviews, etc.
"""

import requests
import time
import json
import sys
from datetime import datetime


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
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")


def print_success(text: str):
    print(f"{Colors.GREEN}[OK]{Colors.RESET} {text}")


def print_error(text: str):
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {text}")

def ask_ollama(prompt, model="llama3.2"):
    """Send a single prompt to Ollama and get response"""
    try:
        response = requests.post("http://localhost:11434/api/generate", 
                               json={
                                   "model": model,
                                   "prompt": prompt,
                                   "stream": False
                               })
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def save_to_json(results, product_descriptions, filename=None):
    """Save results to a JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tag_extraction_results_{timestamp}.json"
    
    # Create structured output
    output_data = {
        "processing_info": {
            "timestamp": datetime.now().isoformat(),
            "total_products": len(results),
            "total_time": sum(r["time"] for r in results),
            "average_time": sum(r["time"] for r in results) / len(results)
        },
        "products": []
    }
    
    for i, (desc, result) in enumerate(zip(product_descriptions, results)):
        product_data = {
            "id": i + 1,
            "description": desc,
            "extracted_tags": result["response"],
            "processing_time": result["time"]
        }
        output_data["products"].append(product_data)
    
    # Save to file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print_success(f"Results saved to: {Colors.BOLD}{filename}{Colors.RESET}")
        return filename
    except Exception as e:
        print_error(f"Error saving to JSON: {e}")
        return None

def process_batch(prompts, model="llama3.2"):
    """Process a list of prompts"""
    results = []

    print(f"\n{Colors.BOLD}Processing {len(prompts)} prompts...{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 50}{Colors.RESET}")

    for i, prompt in enumerate(prompts, 1):
        print(f"{Colors.CYAN}[{i}/{len(prompts)}]{Colors.RESET} Processing...")

        start_time = time.time()
        response = ask_ollama(prompt, model)
        end_time = time.time()

        results.append({
            "prompt": prompt,
            "response": response,
            "time": round(end_time - start_time, 2)
        })

        print_success(f"Done in {Colors.BOLD}{results[-1]['time']}s{Colors.RESET}")
        print(f"  {Colors.DIM}Response: {response[:100]}{'...' if len(response) > 100 else ''}{Colors.RESET}")
        print(f"{Colors.DIM}{'-' * 50}{Colors.RESET}")

    return results

def main():
    # Product descriptions to extract tags from
    product_descriptions = [
        "MacBook Pro 16-inch with M2 chip, 32GB RAM, 1TB SSD storage, Space Gray color, perfect for professional video editing and software development",
        "Wireless Bluetooth headphones with active noise cancellation, 30-hour battery life, premium leather headband, perfect for travel and commuting",
        "Organic cotton t-shirt in navy blue, size medium, eco-friendly sustainable fashion, soft comfortable fabric, casual everyday wear",
        "Stainless steel water bottle, 32oz capacity, double-wall insulated, keeps drinks cold for 24 hours, leak-proof design with sport cap",
        "Gaming mechanical keyboard with RGB backlighting, Cherry MX switches, programmable keys, wired USB connection, black aluminum frame",
        "Electric standing desk with memory presets, adjustable height 28-48 inches, bamboo desktop surface, built-in USB charging ports, anti-collision system",
        "Smart fitness tracker with heart rate monitor, GPS tracking, sleep analysis, waterproof design, 7-day battery life, compatible with iOS and Android",
        "Ceramic non-stick frying pan set, 8-inch and 10-inch sizes, dishwasher safe, oven safe up to 400°F, ergonomic silicone handles",
        "Portable Bluetooth speaker with 360-degree sound, IPX7 waterproof rating, 12-hour battery, voice assistant compatible, compact design for outdoor use",
        "Wool blend winter coat in charcoal gray, size large, tailored fit, water-resistant outer shell, removable hood, inside pockets for phone and wallet",
        "LED desk lamp with adjustable brightness, USB-C charging port, touch controls, eye-care technology, flexible arm design, modern minimalist style",
        "Ergonomic office chair with lumbar support, mesh backrest, adjustable armrests, 360-degree swivel, heavy-duty casters, black fabric upholstery",
        "Vintage leather messenger bag, handcrafted genuine cowhide, laptop compartment fits 15-inch devices, adjustable shoulder strap, antique brass hardware",
        "Instant Pot pressure cooker 6-quart capacity, 7-in-1 functionality, slow cooker, rice cooker, steamer, stainless steel inner pot, recipe book included",
        "Wireless phone charger with fast charging, Qi-compatible, LED indicator light, non-slip surface, works with cases up to 5mm thick, sleek black design"
    ]
    
    # Create consistent prompts for tag extraction
    prompts = []
    for description in product_descriptions:
        prompt = f"Extract 5-7 relevant tags from this product description. Return only the tags separated by commas:\n\n{description}"
        prompts.append(prompt)
    
    # Change this to your installed model
    model_name = "llama3.2"  # or "llama2", "codellama", etc.
    
    # Process all prompts
    results = process_batch(prompts, model_name)
    
    # Save results to JSON
    json_filename = save_to_json(results, product_descriptions)
    
    # Print summary
    print_header("Summary")

    total_time = sum(r["time"] for r in results)
    print(f"  {Colors.BOLD}Total prompts:{Colors.RESET} {len(results)}")
    print(f"  {Colors.BOLD}Total time:{Colors.RESET} {total_time:.2f} seconds")
    print(f"  {Colors.BOLD}Average time:{Colors.RESET} {total_time/len(results):.2f} seconds")

    # Print all results in a clean format
    print(f"\n{Colors.BOLD}{Colors.BLUE}Extracted Tags{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")
    for i, (desc, result) in enumerate(zip(product_descriptions, results), 1):
        print(f"\n{Colors.CYAN}{i}.{Colors.RESET} {Colors.BOLD}Product:{Colors.RESET} {desc[:60]}...")
        print(f"   {Colors.GREEN}Tags:{Colors.RESET} {result['response']}")
        print(f"   {Colors.DIM}Time: {result['time']}s{Colors.RESET}")

if __name__ == "__main__":
    main()