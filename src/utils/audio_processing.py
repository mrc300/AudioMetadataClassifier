import soundfile as sf
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

model_vad = load_silero_vad()

def split_audio_segments(audio_path, sr=16000):
    wav = read_audio(audio_path)
    timestamps = get_speech_timestamps(wav, model_vad, return_seconds=True)

    segments = []
    for i, seg in enumerate(timestamps):
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        segment_wave = wav[start_sample:end_sample]
        temp_filename = f"temp_segment_{i}.wav"
        sf.write(temp_filename, segment_wave.squeeze().numpy(), sr)
        segments.append((temp_filename, start_sample, end_sample))
        
    return segments
