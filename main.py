import torch
import whisper
import soundfile as sf
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import yake
import json
import os




AUDIO_PATH = "audio_files/Danish_language.mp3" 
OUTPUT_JSON = "output_metadata.json"
MODEL_SIZE = "tiny" 



model_vad = load_silero_vad()
whisper_model = whisper.load_model(MODEL_SIZE)
kw_extractor = yake.KeywordExtractor()


wav = read_audio(AUDIO_PATH)
speech_timestamps = get_speech_timestamps(
  wav,
  model_vad,
  return_seconds=True, 
)


metadata = []

for i, seg in enumerate(speech_timestamps):
    sr = 16000  
    start_sample = int(seg['start'] * sr)
    end_sample = int(seg['end'] * sr)
    segment_wave = wav[start_sample:end_sample]
    temp_filename = f"temp_segment_{i}.wav"
    sf.write(temp_filename, segment_wave.squeeze().numpy(), sr)

   
    result = whisper_model.transcribe(temp_filename)
    text = result["text"]
    language = result["language"]

   
    keywords = kw_extractor.extract_keywords(text)

    metadata.append({
        "segment_id": i,
        "start_time_sec": start_sample,
        "end_time_sec": end_sample,
        "language": language,
        "transcription": text.strip(),
        "keywords": [kw[0] for kw in keywords[:5]]
    })

  

    
    os.remove(temp_filename)


with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)


