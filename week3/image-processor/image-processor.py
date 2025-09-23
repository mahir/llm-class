#!/usr/bin/env python3
"""
Batch Image Description Extraction using Ollama
Processes images from a source directory and exports descriptions to JSON
"""

import os
import json
import base64
from pathlib import Path
from datetime import datetime
import argparse
import sys
from typing import List, Dict, Optional
import requests
from PIL import Image
import mimetypes

class OllamaImageProcessor:
    def __init__(self, 
                 model_name: str = "llava:7b", 
                 ollama_host: str = "http://localhost:11434",
                 prompt: str = None):
        """
        Initialize the Ollama Image Processor
        
        Args:
            model_name: Ollama model to use (default: llava:7b)
            ollama_host: Ollama server URL
            prompt: Custom prompt for image description
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.api_url = f"{ollama_host}/api/generate"
        self.default_prompt = prompt or "Describe this image in detail, including objects, people, colors, composition, and mood."
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Convert image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def get_image_info(self, image_path: str) -> Dict:
        """Get basic image information"""
        try:
            with Image.open(image_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "size_bytes": os.path.getsize(image_path)
                }
        except Exception as e:
            print(f"Error getting image info for {image_path}: {e}")
            return {}
    
    def describe_image(self, image_path: str, custom_prompt: str = None) -> Dict:
        """
        Send image to Ollama for description
        
        Args:
            image_path: Path to the image file
            custom_prompt: Optional custom prompt
            
        Returns:
            Dict with description and metadata
        """
        prompt = custom_prompt or self.default_prompt
        
        # Encode image
        base64_image = self.encode_image_to_base64(image_path)
        if not base64_image:
            return {
                "error": "Failed to encode image",
                "description": None
            }
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False
        }
        
        try:
            print(f"Processing: {os.path.basename(image_path)}...")
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=120  # 2 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "description": result.get("response", "").strip(),
                    "model_used": self.model_name,
                    "prompt_used": prompt,
                    "error": None
                }
            else:
                return {
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "description": None
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": "Request timeout - image processing took too long",
                "description": None
            }
        except requests.exceptions.ConnectionError:
            return {
                "error": "Connection error - is Ollama running?",
                "description": None
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {str(e)}",
                "description": None
            }
    
    def get_image_files(self, directory: str) -> List[str]:
        """Get all supported image files from directory"""
        image_files = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def process_directory(self, 
                         source_dir: str, 
                         output_file: str = None,
                         custom_prompt: str = None,
                         recursive: bool = True) -> Dict:
        """
        Process all images in a directory
        
        Args:
            source_dir: Source directory containing images
            output_file: Output JSON file path
            custom_prompt: Custom prompt for all images
            recursive: Process subdirectories recursively
            
        Returns:
            Dict containing processing results
        """
        
        # Get all image files
        if recursive:
            image_files = self.get_image_files(source_dir)
        else:
            directory_path = Path(source_dir)
            image_files = [str(f) for f in directory_path.iterdir() 
                          if f.is_file() and f.suffix.lower() in self.supported_formats]
        
        if not image_files:
            return {
                "error": "No supported image files found",
                "results": [],
                "summary": {}
            }
        
        print(f"Found {len(image_files)} image files to process...")
        
        # Process each image
        results = []
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Get image info
            image_info = self.get_image_info(image_path)
            
            # Get description from Ollama
            description_result = self.describe_image(image_path, custom_prompt)
            
            # Compile result
            result = {
                "file_path": image_path,
                "file_name": os.path.basename(image_path),
                "file_size_bytes": os.path.getsize(image_path),
                "processed_at": datetime.now().isoformat(),
                "image_info": image_info,
                **description_result  # Merge description results
            }
            
            results.append(result)
            
            if result["error"]:
                failed += 1
                print(f"  ❌ Failed: {result['error']}")
            else:
                successful += 1
                print(f"  ✅ Success")
        
        # Compile final results
        final_results = {
            "metadata": {
                "source_directory": source_dir,
                "processed_at": datetime.now().isoformat(),
                "model_used": self.model_name,
                "total_files": len(image_files),
                "successful": successful,
                "failed": failed,
                "custom_prompt": custom_prompt
            },
            "results": results,
            "summary": {
                "total_images": len(image_files),
                "successfully_processed": successful,
                "failed_to_process": failed,
                "success_rate": f"{(successful/len(image_files)*100):.1f}%" if image_files else "0%"
            }
        }
        
        # Save to JSON file
        if output_file:
            self.save_to_json(final_results, output_file)
        
        return final_results
    
    def save_to_json(self, data: Dict, output_file: str):
        """Save results to JSON file"""
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error saving JSON file: {e}")

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Batch process images for description extraction using Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_processor.py /path/to/images
  python image_processor.py /path/to/images -o descriptions.json
  python image_processor.py /path/to/images -m "llava:13b" --prompt "What objects are visible?"
  python image_processor.py /path/to/images --no-recursive -o results.json
        """
    )
    
    parser.add_argument(
        "source_directory",
        help="Source directory containing images"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="image_descriptions.json",
        help="Output JSON file path (default: image_descriptions.json)"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="llava:7b",
        help="Ollama model to use (default: llava:7b)"
    )
    
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--prompt",
        help="Custom prompt for image descriptions"
    )
    
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't process subdirectories recursively"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = OllamaImageProcessor(
            model_name=args.model,
            ollama_host=args.host,
            prompt=args.prompt
        )
        
        # Process directory
        results = processor.process_directory(
            source_dir=args.source_directory,
            output_file=args.output,
            custom_prompt=args.prompt,
            recursive=not args.no_recursive
        )
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Total images: {results['summary']['total_images']}")
        print(f"Successfully processed: {results['summary']['successfully_processed']}")
        print(f"Failed to process: {results['summary']['failed_to_process']}")
        print(f"Success rate: {results['summary']['success_rate']}")
        
        if results['summary']['failed_to_process'] > 0:
            print(f"\nFailed files:")
            for result in results['results']:
                if result['error']:
                    print(f"  - {result['file_name']}: {result['error']}")
        
        return 0 if results['summary']['failed_to_process'] == 0 else 1
        
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())