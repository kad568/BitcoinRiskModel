FROM python:3.11

WORKDIR /app

COPY . .

RUN python -m venv venv

RUN venv/bin/pip install --upgrade pip

RUN venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/app/venv/bin:$PATH"