#!/usr/bin/env python3
"""
ArXiv Research Paper Summarizer using Ollama
Downloads, processes, and summarizes academic papers from arXiv
"""

import os
import json
import requests
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import re
from urllib.parse import urlparse
import PyPDF2
import feedparser
from dataclasses import dataclass

@dataclass
class ArxivPaper:
    """Data class to hold arXiv paper information"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    pdf_url: str
    categories: List[str]
    published: str
    updated: str = None
    doi: str = None

class ArxivSummarizer:
    def __init__(self, 
                 model_name: str = "llama3.1:8b",
                 ollama_host: str = "http://localhost:11434",
                 cache_dir: str = "./arxiv_cache"):
        """
        Initialize the ArXiv Summarizer
        
        Args:
            model_name: Ollama model to use
            ollama_host: Ollama server URL
            cache_dir: Directory to cache downloaded papers
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.api_url = f"{ollama_host}/api/generate"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Summary prompts for different paper sections
        self.prompts = {
            "abstract": """Provide a concise 2-3 sentence summary of this research paper abstract, focusing on the main contribution and key findings:

Abstract: {text}

Summary:""",
            
            "full_paper": """Analyze this research paper and provide a comprehensive summary with the following sections:

**Main Contribution**: What is the primary innovation or finding?
**Methods**: What approach or methodology was used?
**Key Results**: What were the most important findings?
**Significance**: Why is this work important to the field?
**Limitations**: What are the potential weaknesses or limitations?

Paper content: {text}

Summary:""",
            
            "technical": """Provide a technical summary of this research paper for researchers in the field:

**Problem Statement**: What specific problem does this solve?
**Technical Approach**: What methods, algorithms, or techniques were used?
**Experimental Setup**: How was the work validated?
**Quantitative Results**: What are the key metrics and performance numbers?
**Comparison**: How does this compare to existing work?
**Technical Limitations**: What are the technical constraints or assumptions?

Paper content: {text}

Technical Summary:""",
            
            "layman": """Explain this research paper in simple terms that a non-expert could understand:

**What they studied**: What was the research about?
**Why it matters**: Why is this important in everyday terms?
**How they did it**: What did the researchers do (in simple terms)?
**What they found**: What were the main discoveries?
**Real-world impact**: How might this affect people or society?

Paper content: {text}

Simple Explanation:"""
        }
    
    def extract_arxiv_id(self, url_or_id: str) -> str:
        """Extract arXiv ID from URL or return ID if already provided"""
        # Handle direct arXiv ID
        if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', url_or_id):
            return url_or_id
        
        # Handle arXiv URLs
        patterns = [
            r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)',
            r'(\d{4}\.\d{4,5}(?:v\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract arXiv ID from: {url_or_id}")
    
    def fetch_paper_metadata(self, arxiv_id: str) -> ArxivPaper:
        """Fetch paper metadata from arXiv API"""
        # Clean arXiv ID (remove version if present for API call)
        clean_id = re.sub(r'v\d+$', '', arxiv_id)
        api_url = f"http://export.arxiv.org/api/query?id_list={clean_id}"
        
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Extract paper information
            entry = root.find('{http://www.w3.org/2005/Atom}entry')
            if entry is None:
                raise ValueError(f"Paper not found: {arxiv_id}")
            
            # Extract basic information
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            
            # Extract authors
            authors = []
            for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name = author.find('{http://www.w3.org/2005/Atom}name').text
                authors.append(name)
            
            # Extract categories
            categories = []
            for category in entry.findall('{http://www.w3.org/2005/Atom}category'):
                categories.append(category.get('term'))
            
            # Find PDF URL
            pdf_url = None
            for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                if link.get('type') == 'application/pdf':
                    pdf_url = link.get('href')
                    break
            
            if not pdf_url:
                pdf_url = f"http://arxiv.org/pdf/{clean_id}.pdf"
            
            # Extract DOI if available
            doi = None
            doi_element = entry.find('{http://arxiv.org/schemas/atom}doi')
            if doi_element is not None:
                doi = doi_element.text
            
            # Extract updated date if available
            updated = None
            updated_element = entry.find('{http://www.w3.org/2005/Atom}updated')
            if updated_element is not None:
                updated = updated_element.text
            
            return ArxivPaper(
                title=title,
                authors=authors,
                abstract=abstract,
                arxiv_id=arxiv_id,
                pdf_url=pdf_url,
                categories=categories,
                published=published,
                updated=updated,
                doi=doi
            )
            
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch metadata: {e}")
        except ET.ParseError as e:
            raise Exception(f"Failed to parse arXiv response: {e}")
    
    def download_pdf(self, paper: ArxivPaper) -> str:
        """Download PDF and return local file path"""
        pdf_filename = f"{paper.arxiv_id.replace('/', '_')}.pdf"
        pdf_path = self.cache_dir / pdf_filename
        
        # Return cached file if it exists
        if pdf_path.exists():
            print(f"Using cached PDF: {pdf_path}")
            return str(pdf_path)
        
        try:
            print(f"Downloading PDF from {paper.pdf_url}...")
            response = requests.get(paper.pdf_url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"PDF downloaded: {pdf_path}")
            return str(pdf_path)
            
        except requests.RequestException as e:
            raise Exception(f"Failed to download PDF: {e}")
    
    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 20) -> str:
        """Extract text from PDF file"""
        try:
            text_parts = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = min(len(pdf_reader.pages), max_pages)
                
                print(f"Extracting text from {num_pages} pages...")
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Clean up text
                    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
                    text_parts.append(text)
                
                full_text = ' '.join(text_parts)
                print(f"Extracted {len(full_text)} characters")
                return full_text
                
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {e}")
    
    def query_ollama(self, prompt: str, max_tokens: int = 1000, timeout: int = 300) -> str:
        """Send query to Ollama and get response"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3,
                "top_p": 0.9,
                "num_ctx": 8192  # Increase context window
            }
        }
        
        try:
            print(f"Sending request to Ollama (prompt length: {len(prompt)} chars)...")
            response = requests.post(
                self.api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception(f"Ollama request timed out after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama - is it running?")
    
    def chunk_text(self, text: str, max_chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for processing"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 500 chars
                sentence_end = text.rfind('.', end - 500, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap if end < len(text) else len(text)
        
        return chunks
    
    def summarize_chunks(self, chunks: List[str], summary_type: str) -> str:
        """Summarize text chunks and combine results"""
        if len(chunks) == 1:
            # Single chunk - use normal prompt
            prompt = self.prompts[summary_type].format(text=chunks[0])
            return self.query_ollama(prompt, timeout=300)
        
        print(f"Processing {len(chunks)} chunks...")
        chunk_summaries = []
        
        # Create chunk-specific prompt
        chunk_prompt = """Summarize this section of a research paper, focusing on key points, methods, and findings. Be concise but comprehensive:

Section: {text}

Summary:"""
        
        # Summarize each chunk
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}...")
            try:
                prompt = chunk_prompt.format(text=chunk)
                summary = self.query_ollama(prompt, max_tokens=500, timeout=180)
                chunk_summaries.append(summary)
            except Exception as e:
                print(f"Warning: Failed to process chunk {i}: {e}")
                chunk_summaries.append(f"[Chunk {i} processing failed]")
        
        # Combine chunk summaries
        combined_text = "\n\n".join([f"Section {i+1}: {summary}" 
                                   for i, summary in enumerate(chunk_summaries)])
        
        # Create final summary prompt
        final_prompt = self.prompts[summary_type].format(text=combined_text)
        
        print("Generating final summary...")
        return self.query_ollama(final_prompt, max_tokens=1500, timeout=300)
    
    def summarize_paper(self, 
                       arxiv_id: str,
                       summary_type: str = "full_paper",
                       use_full_text: bool = True,
                       max_pages: int = 20,
                       max_text_length: int = 50000) -> Dict:
        """
        Summarize a research paper from arXiv
        
        Args:
            arxiv_id: arXiv ID or URL
            summary_type: Type of summary (abstract, full_paper, technical, layman)
            use_full_text: Whether to download and process full PDF
            max_pages: Maximum pages to process from PDF
            max_text_length: Maximum characters to process from PDF
            
        Returns:
            Dict containing paper info and summary
        """
        print(f"Processing paper: {arxiv_id}")
        
        # Extract and validate arXiv ID
        arxiv_id = self.extract_arxiv_id(arxiv_id)
        
        # Fetch metadata
        paper = self.fetch_paper_metadata(arxiv_id)
        print(f"Found paper: {paper.title}")
        print(f"Authors: {', '.join(paper.authors)}")
        
        # Choose text source
        if use_full_text and summary_type != "abstract":
            # Download and extract full paper text
            pdf_path = self.download_pdf(paper)
            text_content = self.extract_text_from_pdf(pdf_path, max_pages)
            
            # Limit text length to prevent overwhelming Ollama
            if len(text_content) > max_text_length:
                print(f"Text too long ({len(text_content)} chars), truncating to {max_text_length} chars")
                text_content = text_content[:max_text_length]
            
        else:
            # Use just the abstract
            text_content = paper.abstract
        
        # Generate summary using chunking if needed
        print("Generating summary with Ollama...")
        
        try:
            if len(text_content) > 4000:  # Need to chunk
                chunks = self.chunk_text(text_content)
                summary = self.summarize_chunks(chunks, summary_type)
            else:
                # Small enough for single request
                prompt = self.prompts[summary_type].format(text=text_content)
                summary = self.query_ollama(prompt, timeout=300)
                
        except Exception as e:
            # Fallback: try with just abstract if full text fails
            if use_full_text and summary_type != "abstract":
                print(f"Full text processing failed ({e}), falling back to abstract only...")
                prompt = self.prompts["abstract"].format(text=paper.abstract)
                summary = self.query_ollama(prompt, timeout=180)
                summary = f"[Note: Summary based on abstract only due to processing error]\n\n{summary}"
            else:
                raise e
        
        # Compile results
        result = {
            "paper_info": {
                "title": paper.title,
                "authors": paper.authors,
                "arxiv_id": paper.arxiv_id,
                "categories": paper.categories,
                "published": paper.published,
                "updated": paper.updated,
                "doi": paper.doi,
                "pdf_url": paper.pdf_url,
                "abstract": paper.abstract
            },
            "summary_info": {
                "summary_type": summary_type,
                "model_used": self.model_name,
                "used_full_text": use_full_text,
                "processed_at": datetime.now().isoformat(),
                "text_length": len(text_content),
                "max_pages_processed": max_pages if use_full_text else None,
                "was_chunked": len(text_content) > 4000 if use_full_text else False
            },
            "summary": summary
        }
        
        return result
    
    def search_and_summarize(self, 
                           query: str,
                           max_results: int = 5,
                           summary_type: str = "full_paper") -> List[Dict]:
        """
        Search arXiv and summarize multiple papers
        
        Args:
            query: Search query
            max_results: Maximum number of papers to process
            summary_type: Type of summary to generate
            
        Returns:
            List of paper summaries
        """
        print(f"Searching arXiv for: {query}")
        
        # Search arXiv
        search_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
        
        try:
            response = requests.get(search_url, timeout=30)
            response.raise_for_status()
            
            # Parse results
            root = ET.fromstring(response.content)
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            
            if not entries:
                print("No papers found for this query.")
                return []
            
            print(f"Found {len(entries)} papers. Processing...")
            
            results = []
            for i, entry in enumerate(entries, 1):
                try:
                    # Extract arXiv ID from entry
                    id_element = entry.find('{http://www.w3.org/2005/Atom}id')
                    arxiv_url = id_element.text
                    arxiv_id = self.extract_arxiv_id(arxiv_url)
                    
                    print(f"\nProcessing {i}/{len(entries)}: {arxiv_id}")
                    
                    # Summarize paper
                    summary_result = self.summarize_paper(
                        arxiv_id=arxiv_id,
                        summary_type=summary_type,
                        use_full_text=True
                    )
                    
                    results.append(summary_result)
                    
                except Exception as e:
                    print(f"Failed to process {arxiv_id}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            raise Exception(f"Search failed: {e}")
    
    def save_results(self, results: Dict, output_file: str):
        """Save results to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Summarize research papers from arXiv using Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Summary Types:
  abstract    - Quick summary of just the abstract
  full_paper  - Comprehensive summary of the full paper
  technical   - Technical summary for researchers
  layman      - Simple explanation for non-experts

Examples:
  # Summarize a single paper
  python arxiv_summarizer.py 2103.00020
  python arxiv_summarizer.py https://arxiv.org/abs/2103.00020
  
  # Different summary types
  python arxiv_summarizer.py 2103.00020 --type technical
  python arxiv_summarizer.py 2103.00020 --type layman
  
  # Search and summarize multiple papers
  python arxiv_summarizer.py --search "attention mechanism transformer" --max-results 3
  
  # Use different model
  python arxiv_summarizer.py 2103.00020 --model llama3.1:70b
        """
    )
    
    # Main argument - either arXiv ID or search mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "arxiv_id",
        nargs='?',
        help="ArXiv ID or URL to summarize"
    )
    group.add_argument(
        "--search",
        help="Search query for multiple papers"
    )
    
    # Options
    parser.add_argument(
        "--type",
        choices=["abstract", "full_paper", "technical", "layman"],
        default="full_paper",
        help="Type of summary to generate (default: full_paper)"
    )
    
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Ollama model to use (default: llama3.1:8b)"
    )
    
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--cache-dir",
        default="./arxiv_cache",
        help="Directory to cache downloaded papers (default: ./arxiv_cache)"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum results for search mode (default: 5)"
    )
    
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum pages to process from PDF (default: 20)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file to save results"
    )
    
    parser.add_argument(
        "--max-text",
        type=int,
        default=50000,
        help="Maximum characters to process from PDF (default: 50000)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for Ollama requests in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--abstract-only",
        action="store_true",
        help="Only use abstract, don't download full PDF"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize summarizer
        summarizer = ArxivSummarizer(
            model_name=args.model,
            ollama_host=args.host,
            cache_dir=args.cache_dir
        )
        
        if args.search:
            # Search mode
            results = summarizer.search_and_summarize(
                query=args.search,
                max_results=args.max_results,
                summary_type=args.type
            )
            
            # Print results
            print("\n" + "="*80)
            print(f"SEARCH RESULTS: {args.search}")
            print("="*80)
            
            for i, result in enumerate(results, 1):
                print(f"\n--- Paper {i} ---")
                print(f"Title: {result['paper_info']['title']}")
                print(f"Authors: {', '.join(result['paper_info']['authors'])}")
                print(f"arXiv ID: {result['paper_info']['arxiv_id']}")
                print(f"Categories: {', '.join(result['paper_info']['categories'])}")
                print(f"\nSummary:")
                print(result['summary'])
                print("-" * 50)
            
            # Save if requested
            if args.output:
                final_results = {
                    "search_query": args.search,
                    "search_date": datetime.now().isoformat(),
                    "summary_type": args.type,
                    "model_used": args.model,
                    "results": results
                }
                summarizer.save_results(final_results, args.output)
        
        else:
            # Single paper mode
            result = summarizer.summarize_paper(
                arxiv_id=args.arxiv_id,
                summary_type=args.type,
                use_full_text=not args.abstract_only,
                max_pages=args.max_pages,
                max_text_length=args.max_text
            )
            
            # Print result
            print("\n" + "="*80)
            print("PAPER SUMMARY")
            print("="*80)
            print(f"Title: {result['paper_info']['title']}")
            print(f"Authors: {', '.join(result['paper_info']['authors'])}")
            print(f"arXiv ID: {result['paper_info']['arxiv_id']}")
            print(f"Categories: {', '.join(result['paper_info']['categories'])}")
            print(f"Published: {result['paper_info']['published']}")
            
            if result['paper_info']['doi']:
                print(f"DOI: {result['paper_info']['doi']}")
            
            print(f"\nAbstract:")
            print(result['paper_info']['abstract'])
            
            print(f"\n{args.type.upper()} SUMMARY:")
            print("-" * 40)
            print(result['summary'])
            
            # Save if requested
            if args.output:
                summarizer.save_results(result, args.output)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())