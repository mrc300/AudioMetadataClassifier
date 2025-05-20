import whisper
import yake
import io
import os
import logging
import soundfile as sf
import torch
import tempfile
import shutil

# === TEMP FOLDER SETUP ===
CUSTOM_TEMP_DIR = os.path.abspath("custom_tmp")
os.makedirs(CUSTOM_TEMP_DIR, exist_ok=True)

# Redirect all tempfile and subprocess temp usage
tempfile.tempdir = CUSTOM_TEMP_DIR
os.environ["TMPDIR"] = CUSTOM_TEMP_DIR
os.environ["TEMP"] = CUSTOM_TEMP_DIR
os.environ["TMP"] = CUSTOM_TEMP_DIR


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device.upper()}")


whisper_model = whisper.load_model("tiny", device=device)
kw_extractor = yake.KeywordExtractor()

def transcribe_audio_segments(segments, sr=16000):
    metadata = []

    logger.info(f"Transcribing {len(segments)} segments...")

    for i, (wave_tensor, start, end) in enumerate(segments):
        logger.info(f"Processing segment {i+1}/{len(segments)}: {start}s → {end}s")

        try:
        
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

        except Exception as e:
            logger.error(f"Failed to process segment {i}: {e}")
        
      
        temp_files = os.listdir(CUSTOM_TEMP_DIR)
        if temp_files:
            logger.debug(f"Temp files in use after segment {i}: {temp_files}")

    logger.info("Transcription complete.")

    
    try:
        shutil.rmtree(CUSTOM_TEMP_DIR)
        logger.info(f"Cleaned up temporary directory: {CUSTOM_TEMP_DIR}")
    except Exception as cleanup_err:
        logger.warning(f"Could not clean up temp directory: {cleanup_err}")

    return metadata
