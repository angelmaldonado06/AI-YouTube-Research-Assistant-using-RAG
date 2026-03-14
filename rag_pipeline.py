from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from transcript import get_transcript, process


def chunk_transcript(processed_transcript, chunk_size=800, chunk_overlap=100):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks


def create_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )


def create_llm():
    llm = OllamaLLM(
        model="llama3.1",
        temperature=0.3
    )

    return llm


def create_faiss_index(chunks, embedding_model):

    return FAISS.from_texts(chunks, embedding_model)


def retrieve(query, faiss_index, k=4):

    docs = faiss_index.similarity_search(query, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)

    return context


def prepare_video(video_url):

    if not video_url:
        return "", None

    fetched_transcript = get_transcript(video_url)

    if fetched_transcript:
        processed_transcript = process(fetched_transcript)
        chunked_transcript = chunk_transcript(processed_transcript)
        embedding_model = create_embedding_model()
        faiss_index = create_faiss_index(chunked_transcript, embedding_model)

        return processed_transcript, faiss_index

    else:
        return "", None
