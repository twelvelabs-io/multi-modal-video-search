FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg (static build)
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils \
    && curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ffmpeg.tar.xz \
    && cd /tmp && xz -d ffmpeg.tar.xz && tar xf ffmpeg.tar \
    && cp /tmp/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ffmpeg \
    && cp /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ffprobe \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg* \
    && apt-get purge -y curl xz-utils && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

ENV PORT=8000
EXPOSE 8000

CMD ["python3", "app.py"]
