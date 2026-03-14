# AI YouTube Research Assistant using RAG
AI-powered tool that summarizes YouTube videos and answers questions about their content using Retrieval-Augmented Generation (RAG).

## Features
  • Generates summaries of YouTube videos
  • Ask questions about the video and get answers with timestamps

## How it works
### Pipeline:
YouTube URL
    ↓
Transcript Retrieval
    ↓
Text Chunking
    ↓
Embeddings
    ↓
FAISS Vector Search
    ↓
LLM (Llama 3.1)
    ↓
Summary / Question Answering

### Key technologies used:

  • LangChain for prompt orchestration
  • FAISS for retrieval
  • Ollama for running models locally
  • Llama 3.1 for generation
  • HugginFace embeddings for semantic search

## Installation

  1. Clone the repository
  2. Install Python dependencies (pip install -r requirements.txt)
  3. Install Ollama (https://ollama.com/)
  4. Download the model (ollama pull llama3.1)
  5. Run the application (ollama pull llama3.1)

## Project Structure

youtube-rag-assistant
│
├── app.py
├── prompts.py
├── rag_pipeline.py
├── requirements.txt
├── transcript.py
├── README.md
└── .gitignore
