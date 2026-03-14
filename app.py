import gradio as gr
from rag_pipeline import prepare_video, retrieve, create_llm
from prompts import (
    create_summary_prompt,
    create_summary_chain,
    create_qa_prompt_template,
    create_qa_chain
)

processed_transcript = ""
faiss_index = None

def summarize_video(video_url):
    global processed_transcript, faiss_index

    processed_transcript, faiss_index = prepare_video(video_url)
    
    if processed_transcript:
        llm = create_llm()
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        summary = summary_chain.run({
            "transcript" : processed_transcript
        })
        return summary
    else:
        return "No transcripts available"
    

def answer_question(video_url, question):
    global processed_transcript, faiss_index

    if not processed_transcript:
        processed_transcript, faiss_index = prepare_video(video_url)
    
    if processed_transcript and question:
        llm = create_llm()

        context = retrieve(question, faiss_index)
        qa_prompt = create_qa_prompt_template()

        qa_chain = create_qa_chain(llm, qa_prompt)

        answer = qa_chain.run({
            "context" : context,
            "question" : question
        })

        return answer
    else:
        return "No transcript available"


with gr.Blocks() as interface:

    video_url = gr.Textbox(
        label="YouTube Video URL",
        placeholder="Enter YouTube URL"
    )

    summary_output = gr.Textbox(
        label="Video Summary",
        lines=6
    )

    summarize_btn = gr.Button("Summarize Video")

    summarize_btn.click(
        summarize_video,
        inputs=[video_url],
        outputs=[summary_output]
    )

    question_input = gr.Textbox(
        label="Ask a Question"
    )

    answer_output = gr.Textbox(
        label="Answer",
        lines=6
    )

    question_btn = gr.Button("Ask Question")

    question_btn.click(
        answer_question,
        inputs=[video_url, question_input],
        outputs=[answer_output]
    )

interface.launch()