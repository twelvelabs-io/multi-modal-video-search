FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg (apt — has working HTTPS/TLS support unlike static builds)
# Also install static ffmpeg (has libvmaf for QC analysis, but no TLS — local files only)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg xz-utils curl \
    && curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
       | tar -xJ --strip-components=1 -C /usr/local/bin/ --wildcards '*/ffmpeg' \
    && mv /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg-vmaf \
    && apt-get purge -y xz-utils curl && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

ENV PORT=8000
EXPOSE 8000

CMD ["python3", "app.py"]
