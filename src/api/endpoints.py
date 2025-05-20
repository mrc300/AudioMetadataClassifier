from fastapi import APIRouter, UploadFile, File, HTTPException
from src.models.schema import AudioResponse
from src.services.audio_utils import process_audio

router = APIRouter()


@router.post("/process-audio", response_model=AudioResponse)
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp3", ".wav")):
        raise HTTPException(status_code=400, detail="Only .mp3 or .wav files are allowed.")

    try:
        result = await process_audio(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
