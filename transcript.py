from youtube_transcript_api import YouTubeTranscriptApi
import re


def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

 
def get_transcript(url):

    video_id = get_video_id(url)
    ytt_api = YouTubeTranscriptApi()

    transcripts = ytt_api.list(video_id)
    transcript = ""
    for t in transcripts:
        if t.language_code == "en":

            # Prefer manually created transcripts when available
            if not t.is_generated:
                transcript = t.fetch()
                break

            # Use auto-generated transcript only as fallback
            if len(transcript) == 0:
                transcript = t.fetch()

    return transcript if transcript else None


def seconds_to_hhmmss(seconds):

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def process(transcript):
    # Initialize an empty string to hold the formatted transcript
    txt = ""
   
    # Loop through each entry in the transcript
    for i in transcript:
        try:
            # Append the text and its start time to the output string
            hhmmss_time = seconds_to_hhmmss(i.start)
            txt += f"Text: {i.text} Timestamp: {hhmmss_time}\n"
        except KeyError:
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass
           
    # Return the processed transcript as a single string
    return txt