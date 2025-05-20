from fastapi import APIRouter, UploadFile, File
from src.utils.audio_processing import split_audio_segments
from src.services.transcription_service import transcribe_audio_segments
from src.models.schema import TranscriptionResponse
import os

router = APIRouter()

@router.post("/api/generate/" , response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = f"temp_upload_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    segments = split_audio_segments(temp_path)
    metadata = transcribe_audio_segments(segments)

    os.remove(temp_path)
    return {"segments": metadata}
