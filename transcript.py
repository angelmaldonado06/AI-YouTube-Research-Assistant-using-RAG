from youtube_transcript_api import YouTubeTranscriptApi
import re


def get_video_id(url):    
    # Support common YouTube URL formats
    pattern = (
        r"(?:https?:\/\/)?(?:www\.)?"
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/)"
        r"([a-zA-Z0-9_-]{11})"
    )
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


def normalize_transcript_entries(transcript):
    normalized_entries = []

    for entry in transcript:
        text = entry.get("text") if isinstance(entry, dict) else getattr(entry, "text", None)
        start = entry.get("start") if isinstance(entry, dict) else getattr(entry, "start", None)

        if not text or start is None:
            continue

        normalized_entries.append(
            {
                "text": text.strip(),
                "start_seconds": float(start),
                "timestamp": seconds_to_hhmmss(float(start)),
            }
        )

    return normalized_entries


def format_transcript_entries(transcript_entries):
    txt = ""

    for entry in transcript_entries:
        txt += f"Text: {entry['text']} Timestamp: {entry['timestamp']}\n"

    return txt


def process(transcript):
    # Keep backward compatibility with the original app while normalizing first.
    normalized_entries = normalize_transcript_entries(transcript)
    return format_transcript_entries(normalized_entries)
