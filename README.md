# RAG (Retrieval-Augmented Generation)

A hands-on project to learn and implement the core pipeline of Retrieval-Augmented Generation using LangChain, open-source embeddings, and vector databases.

## Why This Project?

RAG is a technique that enhances LLM responses by first retrieving relevant context from your own data. Instead of relying solely on the model's training knowledge, the system fetches the most relevant chunks from a knowledge base and feeds them to the LLM as grounded context. This project walks through each stage of that pipeline from scratch.

## What Was Built

### Document Ingestion

Loaded raw unstructured data into LangChain `Document` objects — the standard format for passing content through a RAG pipeline. Each document carries `page_content` (the text) and `metadata` (source, author, page number, etc.) for filtering and traceability.

Two types of data sources were ingested:

- **Text files** — Sample `.txt` files covering Python programming and machine learning basics, loaded using LangChain's `TextLoader`.
- **PDF files** — A research paper on the impact of AI, processed using PDF parsing libraries (`PyPDF`, `PyMuPDF`) to extract structured text from multi-page documents.

### Text Splitting

Long documents are too large to embed or feed into an LLM in one piece. Text splitters break documents into smaller, overlapping chunks that preserve context at the boundaries. This ensures that no important information is lost at the split points.

### Embedding Generation

Converted each text chunk into a dense vector (embedding) using `sentence-transformers`. These embeddings capture the semantic meaning of the text, enabling similarity-based search rather than keyword matching.

### Vector Store (ChromaDB)

Stored all embeddings in **ChromaDB**, a lightweight vector database. ChromaDB indexes the embeddings and supports fast similarity search — given a query, it retrieves the most relevant document chunks based on cosine similarity.

**FAISS** (Facebook AI Similarity Search) is also included as an alternative vector store for high-performance similarity search at scale.

## Tech Stack

| Component | Purpose |
|---|---|
| **LangChain** | Orchestration framework for the entire RAG pipeline — loading, splitting, embedding, retrieving |
| **Sentence-Transformers** | Generates dense vector embeddings from text using pre-trained models |
| **ChromaDB** | Persistent vector database for storing and querying embeddings |
| **FAISS** | Alternative similarity search library optimized for large-scale vector retrieval |
| **PyPDF / PyMuPDF** | PDF parsing libraries for extracting text from PDF documents |

## Setup

This project uses **uv** for dependency management.

```bash
# Create virtual environment and install dependencies
uv venv
uv add -r requirements.txt
uv add ipykernel
```

Run the notebooks interactively to walk through each stage of the RAG pipeline.
