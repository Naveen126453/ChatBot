# Document-Based AI Chatbot (RAG)

## Overview
This project is a document-based AI chatbot built using Flask, LangChain, FAISS, and open-source LLMs via Ollama.  
It follows the Retrieval-Augmented Generation (RAG) architecture to answer queries from PDF documents.

## Tech Stack
- Python
- Flask
- LangChain
- FAISS
- HuggingFace Embeddings
- Ollama (Mistral / Phi)

## Workflow
1. PDFs are loaded and split into chunks
2. Embeddings are generated using sentence-transformers
3. Vectors are stored in FAISS
4. User queries retrieve relevant chunks
5. LLM generates grounded responses

## How to Run
```bash
python ingest.py
python app.py

