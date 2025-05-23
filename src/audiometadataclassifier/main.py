from fastapi import FastAPI
from src.audiometadataclassifier.api.endpoints import router

app = FastAPI(title="Audio Metadata Classifier")
app.include_router(router)
