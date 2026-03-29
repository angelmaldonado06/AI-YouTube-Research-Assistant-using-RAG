# AI YouTube Research Assistant using RAG
AI-powered tool that summarizes YouTube videos and answers questions about their content using Retrieval-Augmented Generation (RAG).


## Features
- Generates summaries of YouTube videos
- Answers questions about the video using transcript-based retrieval
- Preserves transcript timestamps inside retrieved chunks


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
- LangChain for prompt orchestration
- FAISS for retrieval
- Ollama for running models locally
- Llama 3.1 for generation
- Hugging Face embeddings for semantic search


## Installation

- Clone the repository
- Install Python dependencies with `pip install -r requirements.txt`
- Install Ollama from [ollama.com](https://ollama.com/)
- Download the model with `ollama pull llama3.1`
- Run the model with `ollama run llama3.1`
- Start the app with `python app.py`


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
