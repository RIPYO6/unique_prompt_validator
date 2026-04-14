# 🚀 LLM Prompt Uniqueness Validator

Ensuring semantic diversity and preventing redundancy in LLM instruction datasets. This tool provides a multi-stage validation engine to audit new prompts against existing datasets using exact matching and semantic vector search.

## ✨ Features

- **Hybrid Validation Engine**: Combines O(1) exact-match lookups with semantic similarity search.
- **Vector Search Cache**: Powered by **ChromaDB** and **LangChain** for fast, local similarity analysis.
- **Ollama Integration**: Uses locally hosted embeddings (Gemma) via Ollama for privacy and offline capability.
- **Premium Gradio UI**: Interactive dashboard with real-time similarity scoring and color-coded warning levels.
- **Human-in-the-Loop**: Seamless workflow for reviewing candidates and persisting approved prompts to the database.

## 🛠️ Tech Stack

- **Logic**: Python 3.10+
- **LLM Framework**: LangChain
- **Embeddings**: Ollama (`embeddinggemma:300m`)
- **Vector Store**: ChromaDB
- **UI/UX**: Gradio
- **Data**: [10k Prompts Ranked](https://huggingface.co/datasets/data-is-better-together/10k_prompts_ranked) (Hugging Face)

## 🚀 Getting Started

### Prerequisites

1.  **Ollama**: Install [Ollama](https://ollama.ai/) and pull the embedding model:
    ```bash
    ollama pull embeddinggemma:300m
    ```

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/RIPYO6/unique-prompt-validator.git
    cd unique-prompt-validator
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Run the validator dashboard:
```bash
python app.py
```
Open the URL provided by Gradio (usually `http://127.0.0.1:7860`) in your browser.

## 📊 How it Works

1.  **Exact Match**: The system first checks a hash-set of all existing prompts for an instantaneous duplicate check.
2.  **Semantic Search**: If no exact match is found, it generates a vector embedding for the candidate and queries the ChromaDB collection for the top 5 most similar existing prompts.
3.  **Scoring**: Similarity is calculated via cosine distance.
    - **Red**: > 85% similarity (Hard Reject)
    - **Yellow**: 70% - 85% similarity (Potential Redundancy)
    - **Green**: < 70% similarity (Unique)

