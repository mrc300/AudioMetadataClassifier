from pydantic import BaseModel, Field
from typing import List

class SegmentMetadata(BaseModel):
    segment_id: int = Field(..., description="The index of the audio segment")
    start_time_sec: float = Field(..., description="Start time of the segment in seconds")
    end_time_sec: float = Field(..., description="End time of the segment in seconds")
    language: str = Field(..., description="Detected spoken language (ISO code)")
    transcription: str = Field(..., description="Transcribed text from the audio segment")
    keywords: List[str] = Field(..., description="Top 5 extracted keywords")

class TranscriptionResponse(BaseModel):
    segments: List[SegmentMetadata] = Field(..., description="List of all processed audio segments")
