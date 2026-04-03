# AI YouTube Research Assistant using RAG

AI-powered tool that summarizes YouTube videos and answers questions about their content using Retrieval-Augmented Generation (RAG).

## Features

- Generates summaries of YouTube videos
- Answers questions about the video using transcript-based retrieval
- Preserves transcript timestamps inside retrieved chunks
- Evaluates the RAG pipeline with RAGAS

## How It Works

### Pipeline

```text
YouTube URL
  -> Transcript Retrieval
  -> Text Chunking
  -> Embeddings
  -> FAISS Vector Search
  -> LLM (Llama 3.1)
  -> Summary / Question Answering
  -> RAG Evaluation (RAGAS)
```

### Key Technologies

- LangChain for prompt orchestration
- FAISS for retrieval
- Ollama for running models locally
- Llama 3.1 for generation
- Hugging Face embeddings for semantic search
- RAGAS for evaluation

## Installation

- Clone the repository
- Install Python dependencies with `pip install -r requirements.txt`
- Install Ollama from [ollama.com](https://ollama.com/)
- Download the model with `ollama pull llama3.1`
- Run the model with `ollama run llama3.1`
- Start the app with `python app.py`

## Evaluation With RAGAS

Evaluation is the step where you measure whether your RAG system is actually performing well. In a RAG application, this is important because a fluent answer can still be wrong, ungrounded, or based on poor retrieval.

This project now includes `evaluation.py`, which:

1. Loads a YouTube transcript and creates the vector index
2. Runs a set of evaluation questions through the same RAG pipeline
3. Collects the retrieved contexts and generated answers
4. Scores the results with RAGAS

### Metrics Used

- `faithfulness`: checks whether the answer is supported by the retrieved context
- `answer_relevancy`: checks whether the answer actually addresses the question
- `context_precision`: checks whether the retrieved chunks are useful rather than noisy
- `context_recall`: checks whether retrieval found enough of the needed information

### Create an Evaluation Dataset

Use `sample_eval_dataset.json` as a template. Each item should include:

- `question`: a realistic question a user might ask
- `ground_truth`: the reference answer you expect from the transcript

Example:

```json
[
  {
    "question": "What is the speaker's main argument?",
    "ground_truth": "The speaker argues that..."
  }
]
```

### Run Evaluation

```bash
python evaluation.py --video-url "YOUTUBE_URL_HERE" --dataset sample_eval_dataset.json
```

### How To Explain This In An Interview

You can say:

`I added a RAGAS-based evaluation step so I can measure both retrieval quality and answer quality. I use metrics like faithfulness, answer relevancy, context precision, and context recall to detect hallucinations, weak retrieval, and grounding problems before changing prompts or models.`

## Project Structure

```text
youtube-rag-assistant
|
|-- app.py
|-- evaluation.py
|-- prompts.py
|-- rag_pipeline.py
|-- requirements.txt
|-- sample_eval_dataset.json
|-- transcript.py
`-- README.md
```
