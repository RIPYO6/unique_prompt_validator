import os
import pandas as pd
from datasets import load_dataset
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Tuple

class PromptValidator:
    def __init__(self, db_path: str = "./chroma_db", model_name: str = "embeddinggemma:300m"):
        self.db_path = db_path
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        self.vector_db = None
        self.existing_prompts_set = set()
        
        # Initialize the system
        self._initialize_system()

    def _normalize(self, text: str) -> str:
        return text.strip().lower()

    def _initialize_system(self):
        print("Initializing Prompt Uniqueness Validator...")
        
        # Load the vector store if it exists
        if os.path.exists(self.db_path) and len(os.listdir(self.db_path)) > 0:
            print(f"Loading existing ChromaDB from {self.db_path}...")
            self.vector_db = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            all_docs = self.vector_db.get()
            self.existing_prompts_set = {self._normalize(doc) for doc in all_docs['documents']}
            print(f"Loaded {len(self.existing_prompts_set)} prompts from cache.")
        else:
            print("Building new ChromaDB from Hugging Face dataset...")
            print("Fetching 'data-is-better-together/10k_prompts_ranked' from Hugging Face...")
            ds = load_dataset("data-is-better-together/10k_prompts_ranked", split="train")
            
            prompts = [p for p in ds["prompt"] if p]
            print(f"Total prompts found: {len(prompts)}")
            
            # Normalize and build the exact match set
            self.existing_prompts_set = {self._normalize(p) for p in prompts}
            
            # For the vector DB, we'll index everything
            print(f"Indexing ALL {len(prompts)} prompts into semantic search engine...")
            documents = [Document(page_content=p) for p in prompts]
            
            print("Starting embedding generation (this will take a while for 10k prompts)...")
            self.vector_db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.db_path,
                collection_metadata={"hnsw:space": "cosine"}
            )
            print(f"System initialized. Exact match set size: {len(self.existing_prompts_set)}")
            print(f"Semantic search fully indexed: {len(prompts)} prompts.")

    def check_exact_match(self, prompt: str) -> bool:
        normalized = self._normalize(prompt)
        return normalized in self.existing_prompts_set

    def check_semantic_similarity(self, prompt: str, top_k: int = 5) -> List[Tuple[str, float]]:
        import time
        start_time = time.time()
        print(f"Starting semantic search for prompt: {prompt[:50]}...")
        
        # This call handles both embedding the query and searching the DB
        results = self.vector_db.similarity_search_with_score(prompt, k=top_k)
        
        end_time = time.time()
        print(f"Semantic search completed in {end_time - start_time:.2f} seconds.")
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append((doc.page_content, score))
            
        return formatted_results

    def add_prompt(self, prompt: str):
        normalized = self._normalize(prompt)
        if normalized not in self.existing_prompts_set:
            self.existing_prompts_set.add(normalized)
            self.vector_db.add_documents([Document(page_content=prompt)])
            print(f"Prompt added: {prompt[:50]}...")
        else:
            print("Prompt already exists in set.")

if __name__ == "__main__":
    # Test initialization
    validator = PromptValidator()
    test_prompt = "Provide a detailed workout routine for a beginner."
    print(f"Exact match for '{test_prompt}': {validator.check_exact_match(test_prompt)}")
    
    sim_results = validator.check_semantic_similarity(test_prompt)
    print("\nSemantic search results:")
    for p, s in sim_results:
        print(f"Score: {s:.4f} | Prompt: {p[:100]}...")
