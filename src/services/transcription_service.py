import whisper
import os
import yake

whisper_model = whisper.load_model("tiny")
kw_extractor = yake.KeywordExtractor()

def transcribe_audio_segments(segments):
    metadata = []

    for i, (filename, start, end) in enumerate(segments):
        result = whisper_model.transcribe(filename)
        text = result["text"]
        language = result["language"]
        keywords = kw_extractor.extract_keywords(text)

        metadata.append({
            "segment_id": i,
            "start_time_sec": start,
            "end_time_sec": end,
            "language": language,
            "transcription": text.strip(),
            "keywords": [kw[0] for kw in keywords[:5]]
        })

        os.remove(filename)

    return metadata
