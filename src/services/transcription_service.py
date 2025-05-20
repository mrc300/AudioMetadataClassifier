import whisper
import yake
import io
import os
import logging
import soundfile as sf
import torch
import tempfile
import shutil


CUSTOM_TEMP_DIR = os.path.abspath("custom_tmp")
os.makedirs(CUSTOM_TEMP_DIR, exist_ok=True)


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
          
            if hasattr(wave_tensor, "numpy"):
                wave_data = wave_tensor.squeeze().numpy()
            else:
                wave_data = wave_tensor.squeeze()

          
            audio = whisper.pad_or_trim(wave_data)
            mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

            
            _, probs = whisper_model.detect_language(mel)
            language = max(probs, key=probs.get)

            options = whisper.DecodingOptions()
            result = whisper.decode(whisper_model, mel, options)

            text = result.text.strip()
            keywords = kw_extractor.extract_keywords(text)

            metadata.append({
                "segment_id": i,
                "start_time_sec": start,
                "end_time_sec": end,
                "language": language,
                "transcription": text,
                "keywords": [kw[0] for kw in keywords[:5]]
            })

        except Exception as e:
            logger.error(f"Failed to process segment {i}: {e}")

    logger.info("Transcription complete.")
    return metadata
