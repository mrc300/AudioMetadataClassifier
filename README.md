# Audio Metadata Classifier API

A FastAPI-powered REST API for transcribing audio files, extracting speech segments using VAD (Voice Activity Detection), detecting language, and extracting keywords for metadata enhacement. This project aims to evolve into a robust, scalable, and intelligent audio metadata processing service.

---

## Features

- **Automatic Speech Segmentation** using [Silero VAD](https://github.com/snakers4/silero-vad)
- **Transcription** with [OpenAI Whisper](https://github.com/openai/whisper)
- **Language Detection**
- **Keyword Extraction** via [YAKE](https://github.com/LIAAD/yake)
- **FastAPI** + **Docker** + **Poetry**

---

## Tech Stack

- Python 3.12+
- FastAPI
- Whisper (OpenAI)
- Silero VAD
- YAKE
- Torch (CUDA enabled)
- Docker
- Poetry

---

## Run Locally (Poetry)

```bash
# Install dependencies
poetry install

# Run the API
poetry run uvicorn src.audiometadataclassifier.main:app --reload

# Run the API using script
poetry run start
```

---

## Run in Docker

```bash
# Build the image
docker build -t amc-api .

# Run the container (with GPU if available)
docker run --gpus all -p 8000:8000 amc-api
```

Then go to [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

---

## API Endpoint

**POST** `/api/generate/`

| Field | Type | Description         |
|-------|------|---------------------|
| file  | file | Audio file (e.g., .mp3, .wav) |

### Response:

In JSON:

```json
[
  {
    "segment_id": 0,
    "start_time_sec": 0.0,
    "end_time_sec": 5.4,
    "language": "en",
    "transcription": "Hello, how are you?",
    "keywords": ["hello", "how", "are", "you"]
  }
]
```

---

### Environment Notes

- Make sure `ffmpeg` is installed (handled inside Docker)
- CUDA GPU required for best performance (`torch.cuda.is_available()` should return `True`)
- Whisper model defaults to `tiny`. Can be changed in code

---

## Goals & Future Improvements

---

### Efficient Transcription

- **Batch Transcription**  
  Support for processing multiple audio files or long files split into segments with reduced overhead.

- **Dynamic Model Selection**  
  Automatically choose the optimal Whisper model (`tiny`, `base`, `small`, etc.) based on:
  - Audio duration
  - Language complexity
  - Desired speed vs. accuracy
  - Cost efficiency

---

### Smarter Keyword Extraction

- **ML-powered Keyword Extraction**  
  Replace YAKE with AI-based models:
  - OpenAI embeddings + clustering
  - KeyBERT with transformer backends
  - Transformer-based summarization models (BART, T5)

---

### Audio File Filtering & Preprocessing

- **Input Quality Validation**  
  Analyze:
  - Duration
  - Sampling rate
  - Volume/loudness
  - Silence percentage

- **Preprocessing Enhancements**
  - Normalize volume
  - Trim silent portions
  - Reject or warn on poor quality

---

### Metadata Outputs & Integrations

- **Flexible Output Formats**  
  - JSON
  - CSV
 

---

### API-Level Enhancements

- Rate Limiting & Caching
- Authentication (API keys, OAuth2)
- OpenAPI Client SDK Generation


## Author

**Manuel Cunha**  
[manelfcunha@gmail.com](mailto:manelfcunha@gmail.com)
