from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain


def create_summary_prompt():

    template = """
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "Timestamp" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    Video content:
    {transcript}

    Summary:
    """

    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )

    return prompt


def create_summary_chain(llm, prompt, verbose=True):
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)


def create_qa_prompt_template():
   
    qa_template = """
    You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:

    1. Precise and free from repetition
    2. Consistent with the information provided in the video
    3. Well-organized and easy to understand
    4. Focused on addressing the user's question directly
    5. Mention at most ONE timestamp from the context.
    6. Always place the timestamp at the end of the answer.

    Only answer using the provided context.
    If the answer is not in the context, say "The video does not mention this."

    Note: In the transcript, "Text" refers to the spoken words in the video, timestamp indicated when that part begins in the video.
 
    Relevant Video Context: {context}
    Based on the above context, please answer the following question: {question}
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    return prompt_template


def create_qa_chain(llm, prompt_template, verbose=True):
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)

