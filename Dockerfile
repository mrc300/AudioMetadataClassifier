FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime


RUN apt-get update && apt-get install -y \
    ffmpeg build-essential curl git && apt-get clean

WORKDIR /app

ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry


COPY pyproject.toml ./


RUN poetry config virtualenvs.create false \
 && poetry install --no-root --no-ansi --no-interaction --no-dev


COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.audiometadataclassifier.main:app", "--host", "0.0.0.0", "--port", "8000"]
