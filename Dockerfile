FROM python:3.11-slim

WORKDIR /app

# system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg v4l-utils && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501

ENV STREAMLIT_SERVER_RUN_ON_SAVE=false
CMD ["streamlit", "run", "smart_classroom/dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
