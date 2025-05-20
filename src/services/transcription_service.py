import whisper
import yake
import io
import logging
import soundfile as sf


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


whisper_model = whisper.load_model("tiny", "cuda")
kw_extractor = yake.KeywordExtractor()

def transcribe_audio_segments(segments, sr=16000):
 
    metadata = []

    logger.info(f"Transcribing {len(segments)} segments...")

    for i, (wave_tensor, start, end) in enumerate(segments):
        logger.info(f"Processing segment {i+1}/{len(segments)}: {start}s → {end}s")

        
        buffer = io.BytesIO()
        sf.write(buffer, wave_tensor.squeeze().numpy(), sr, format='WAV')
        buffer.seek(0)

        result = whisper_model.transcribe(buffer)
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

    logger.info("Transcription complete.")
    return metadata
