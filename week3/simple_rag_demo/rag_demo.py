#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) Demo with Ollama
============================================================

This demo implements a basic RAG system for a company FAQ chatbot.

RAG combines three key components:
1. RETRIEVAL: Finding relevant documents from a knowledge base
2. AUGMENTATION: Adding retrieved context to the user's query  
3. GENERATION: Using an LLM to create a contextually-aware response

The demo uses:
- TF-IDF vectorization for document similarity (simple but effective)
- Ollama for local LLM inference (privacy-friendly, no API keys needed)
- A sample knowledge base of company policies and procedures

Author: RAG Demo
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

class SimpleRAG:
    """
    A simple RAG (Retrieval-Augmented Generation) system.
    
    This class demonstrates the core RAG workflow:
    1. Store documents in a searchable format (TF-IDF vectors)
    2. Retrieve relevant documents for user queries
    3. Generate contextually-aware responses using retrieved information
    """
    
    def __init__(self, ollama_model="llama3.2:latest"):
        """
        Initialize the RAG system.
        
        Args:
            ollama_model (str): Name of the Ollama model to use for generation
        """
        # The LLM model we'll use for generating responses
        self.ollama_model = ollama_model
        
        # TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer
        # This converts text into numerical vectors for similarity comparison
        # - stop_words='english': Remove common words like "the", "and", "is"
        # - ngram_range=(1,2): Consider both single words and 2-word phrases
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        
        # Storage for our knowledge base
        self.documents = []        # Original documents with titles and content
        self.doc_vectors = None    # TF-IDF vector representations of documents
        
        # Ollama API endpoint for generating responses
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def add_documents(self, documents):
        """
        Add documents to the knowledge base and create searchable vectors.
        
        This is the "indexing" phase of RAG - we preprocess all documents
        and convert them into a format that allows fast similarity search.
        
        Args:
            documents (list): List of dicts with 'title' and 'content' keys
        """
        # Store the original documents for later retrieval
        self.documents = documents
        
        # Extract just the text content for vectorization
        doc_texts = [doc['content'] for doc in documents]
        
        # Create TF-IDF vectors for all documents
        # This converts each document into a numerical vector where:
        # - Each dimension represents a word or phrase
        # - Values indicate how important that word is to this document
        self.doc_vectors = self.vectorizer.fit_transform(doc_texts)
        
        print(f"Added {len(documents)} documents to knowledge base")
        print(f"Vectorizer learned {len(self.vectorizer.vocabulary_)} unique terms")
    
    def retrieve(self, query, top_k=2):
        """
        Retrieve the most relevant documents for a given query.
        
        This is the "Retrieval" step in RAG. We:
        1. Convert the user's query into the same vector space as our documents
        2. Calculate similarity scores between query and all documents
        3. Return the most similar documents
        
        Args:
            query (str): User's question or search query
            top_k (int): Maximum number of documents to retrieve
            
        Returns:
            list: Most relevant documents with similarity scores
        """
        # If no documents loaded, can't retrieve anything
        if not self.documents:
            return []
        
        # Convert the user's query into a TF-IDF vector using the same
        # vocabulary and weighting scheme as our document vectors
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all documents
        # Cosine similarity measures the angle between vectors:
        # - 1.0 = identical direction (very similar)
        # - 0.0 = perpendicular (unrelated)
        # - -1.0 = opposite direction (opposite meaning)
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Find the indices of the most similar documents
        # np.argsort returns indices sorted by similarity (lowest to highest)
        # [-top_k:] takes the last (highest) k values
        # [::-1] reverses to get highest to lowest
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build list of retrieved documents with metadata
        retrieved_docs = []
        for idx in top_indices:
            # Only include documents above a minimum similarity threshold
            # This prevents retrieving completely unrelated documents
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                retrieved_docs.append({
                    'content': self.documents[idx]['content'],
                    'title': self.documents[idx]['title'],
                    'similarity': similarities[idx]  # For debugging/transparency
                })
        
        return retrieved_docs
    
    def generate_with_ollama(self, prompt):
        """
        Generate a response using the Ollama LLM.
        
        This is the "Generation" step in RAG. We send the user's query
        along with retrieved context to the LLM for a final response.
        
        Args:
            prompt (str): The complete prompt including context and question
            
        Returns:
            str: Generated response from the LLM
        """
        # Prepare the request payload for Ollama's API
        payload = {
            "model": self.ollama_model,     # Which model to use
            "prompt": prompt,               # Our crafted prompt with context
            "stream": False,                # Get complete response, not streaming
            "options": {
                "temperature": 0.7,         # Creativity level (0.0-1.0)
                "top_p": 0.9               # Nucleus sampling parameter
            }
        }
        
        try:
            # First, verify the model exists by checking available models
            models_response = requests.get("http://localhost:11434/api/tags")
            if models_response.status_code == 200:
                available_models = [model['name'] for model in models_response.json()['models']]
                if self.ollama_model not in available_models:
                    return f"Model '{self.ollama_model}' not found. Available models: {', '.join(available_models)}"
            
            # Send the generation request to Ollama
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Extract and return the generated text
            return response.json()['response']
            
        except requests.exceptions.RequestException as e:
            # Provide helpful error messages for common issues
            if "404" in str(e):
                return f"Error: Ollama API endpoint not found. Make sure Ollama is running with 'ollama serve'. Details: {e}"
            return f"Error connecting to Ollama: {e}"
    
    def query(self, question):
        """
        Main RAG pipeline: Retrieve relevant docs and generate response.
        
        This orchestrates the complete RAG workflow:
        1. RETRIEVE: Find relevant documents from knowledge base
        2. AUGMENT: Create a prompt with retrieved context
        3. GENERATE: Use LLM to create contextually-aware response
        
        Args:
            question (str): User's question
            
        Returns:
            str: Generated answer based on retrieved context
        """
        print(f"\n🔍 Query: {question}")
        
        # STEP 1: RETRIEVAL
        # Search our knowledge base for relevant documents
        retrieved_docs = self.retrieve(question)
        
        # If no relevant documents found, return a helpful message
        if not retrieved_docs:
            print("❌ No relevant documents found")
            return "I don't have information about that topic in my knowledge base."
        
        # Show user what documents were retrieved (transparency)
        print(f"📚 Retrieved {len(retrieved_docs)} relevant documents:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"   {i}. {doc['title']} (similarity: {doc['similarity']:.3f})")
        
        # STEP 2: AUGMENTATION  
        # Combine retrieved documents into context for the LLM
        # Format: "Document: Title\nContent\n\nDocument: Title\nContent..."
        context = "\n\n".join([f"Document: {doc['title']}\n{doc['content']}" 
                              for doc in retrieved_docs])
        
        # STEP 3: GENERATION
        # Create a carefully crafted prompt that includes:
        # - Role instruction (you are a customer service assistant)
        # - Context from retrieved documents  
        # - The user's specific question
        # - Instructions for handling missing information
        prompt = f"""You are a helpful customer service assistant for TechFlow Software. 
Use the following context to answer the user's question. If the context doesn't contain 
enough information, say so clearly and don't make up information.

Context:
{context}

Question: {question}

Answer:"""
        
        print("🤖 Generating response using retrieved context...")
        
        # Send the augmented prompt to the LLM for final response generation
        response = self.generate_with_ollama(prompt)
        return response

def create_sample_knowledge_base():
    """
    Create a sample knowledge base for TechFlow Software.
    
    In a real implementation, this might load from:
    - A database of support articles
    - Markdown files in a documentation repository  
    - PDFs of policy documents
    - Web scraped FAQ pages
    - Customer support ticket resolutions
    
    Returns:
        list: Sample documents with titles and content
    """
    return [
        {
            "title": "Return Policy - Enterprise Licenses",
            "content": "TechFlow Software offers a 30-day return policy for enterprise licenses. Full refunds are available if the software hasn't been deployed to more than 5 users and no custom integrations have been implemented. Contact support@techflow.com with your license key for processing."
        },
        {
            "title": "Remote Work Policy", 
            "content": "Employees can work remotely up to 3 days per week. Remote work requires manager approval and completion of the remote work agreement form. All remote workers must be available during core business hours (10 AM - 3 PM EST) and attend mandatory team meetings."
        },
        {
            "title": "Password Reset Instructions",
            "content": "To reset your password: 1) Go to login.techflow.com, 2) Click 'Forgot Password', 3) Enter your email address, 4) Check your email for reset link (may take up to 10 minutes), 5) Create new password with at least 8 characters including numbers and symbols."
        },
        {
            "title": "Product Pricing - Standard Edition",
            "content": "TechFlow Standard Edition: $49/month per user for teams up to 10 users. Includes basic analytics, standard support, and 10GB storage. Annual billing available with 20% discount. Free 14-day trial includes all features."
        },
        {
            "title": "Product Pricing - Enterprise Edition", 
            "content": "TechFlow Enterprise Edition: $99/month per user, minimum 25 users. Includes advanced analytics, priority support, unlimited storage, custom integrations, and dedicated account manager. Custom pricing for 100+ users."
        },
        {
            "title": "System Requirements",
            "content": "Minimum requirements: Windows 10/macOS 10.15/Ubuntu 18.04, 8GB RAM, 2GB disk space, internet connection. Recommended: 16GB RAM, SSD storage. Browser requirements: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+."
        },
        {
            "title": "Vacation Policy",
            "content": "Full-time employees receive 15 days PTO annually, increasing to 20 days after 3 years and 25 days after 5 years. Vacation requests must be submitted at least 2 weeks in advance. Maximum 5 consecutive days without manager approval."
        }
    ]

def main():
    """
    Main function to run the RAG demo.
    
    This function:
    1. Initializes the RAG system
    2. Loads sample documents into the knowledge base
    3. Provides example queries users can try
    4. Runs an interactive chat loop
    """
    print("🚀 TechFlow Software RAG Chatbot Demo")
    print("=" * 50)
    print("This demo shows how RAG (Retrieval-Augmented Generation) works:")
    print("1. 🔍 RETRIEVE: Find relevant documents from knowledge base")
    print("2. 🔗 AUGMENT: Add context to your question")  
    print("3. 🤖 GENERATE: Create answer using LLM + context")
    
    # Initialize the RAG system with default model
    # You can change the model name here if needed (e.g., "llama2", "mistral")
    rag = SimpleRAG(ollama_model="llama3.2:latest")
    
    # Load our sample knowledge base
    print("\n📖 Loading knowledge base...")
    documents = create_sample_knowledge_base()
    rag.add_documents(documents)
    
    # Show users what kinds of questions they can ask
    sample_queries = [
        "What's your return policy for enterprise licenses?",
        "How do I reset my password?",
        "What are the pricing options for your software?", 
        "Can I work from home?",
        "What are the system requirements?",
        "How much vacation time do I get?"
    ]
    
    print(f"\n🎯 Sample queries you can try:")
    for i, query in enumerate(sample_queries, 1):
        print(f"   {i}. {query}")
    
    print(f"\n💡 You can also ask your own questions about TechFlow Software!")
    print(f"📋 Available topics: pricing, returns, remote work, passwords, system requirements, vacation")
    print(f"🔍 Watch how the system retrieves relevant documents before generating answers!")
    
    # Interactive chat loop
    while True:
        print("\n" + "-" * 50)
        question = input("❓ Enter your question (or 'quit' to exit): ").strip()
        
        # Handle exit commands
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        # Skip empty inputs
        if not question:
            continue
            
        # Process the user's query through our RAG pipeline
        try:
            answer = rag.query(question)
            print(f"\n💬 Answer: {answer}")
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print("🔧 Try checking if Ollama is running: 'ollama serve'")

if __name__ == "__main__":
    """
    Entry point for the script.
    
    Before running, make sure you have:
    1. Ollama installed and running ('ollama serve')
    2. A model pulled ('ollama pull llama3.2' or similar)
    3. Required Python packages installed
    """
    
    # Check if required packages are installed
    try:
        import sklearn
        import numpy as np
        import requests
        print("✅ All required packages found")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("📦 Install with: pip install scikit-learn numpy requests")
        exit(1)
    
    # Start the demo
    main()