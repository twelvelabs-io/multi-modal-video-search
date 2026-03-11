"""Backfill video fingerprints and thumbnails for existing indexed videos."""
import os
import sys
import subprocess
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from dotenv import load_dotenv
load_dotenv()

from src.mongodb_client import MongoDBEmbeddingClient
from src.compare_client import compute_fingerprint

S3_BUCKET = os.environ.get("S3_BUCKET", "")

def generate_thumbnail(s3_client, bucket, proxy_key):
    """Download video, extract thumbnail with ffmpeg, upload back to S3."""
    thumbnail_key = proxy_key.rsplit(".", 1)[0] + "_thumb.jpg"
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            s3_client.download_file(bucket, proxy_key, tmp_video.name)
            tmp_video_path = tmp_video.name

        tmp_thumb_path = tmp_video_path.rsplit(".", 1)[0] + "_thumb.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_video_path,
            "-ss", "2", "-vframes", "1",
            "-vf", "scale=480:-1", "-q:v", "3",
            tmp_thumb_path
        ], capture_output=True, timeout=30)

        if os.path.exists(tmp_thumb_path) and os.path.getsize(tmp_thumb_path) > 0:
            s3_client.upload_file(
                tmp_thumb_path, bucket, thumbnail_key,
                ExtraArgs={"ContentType": "image/jpeg"}
            )
            os.unlink(tmp_video_path)
            os.unlink(tmp_thumb_path)
            return thumbnail_key
        else:
            os.unlink(tmp_video_path)
            return None
    except Exception as e:
        print(f"    Thumbnail failed: {e}")
        return None


def main():
    mongodb = MongoDBEmbeddingClient(connection_string=os.environ.get("MONGODB_URI"))
    s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    collection = mongodb.db["unified-embeddings"]
    video_ids = collection.distinct("video_id")
    print(f"Found {len(video_ids)} videos to process")

    created = 0
    skipped = 0
    thumbs = 0
    for vid in video_ids:
        existing = mongodb.get_video_fingerprint(vid)
        if existing and existing.get("thumbnail_key"):
            skipped += 1
            continue

        segments = mongodb.get_segments_for_video(vid)
        if not segments:
            print(f"  SKIP {vid}: no segments found")
            skipped += 1
            continue

        fp = compute_fingerprint(segments)

        video_name = vid
        proxy_key = None
        for seg in segments:
            s3_uri = seg.get("s3_uri", "")
            if s3_uri:
                video_name = s3_uri.split("/")[-1]
                proxy_key = s3_uri.replace(f"s3://{S3_BUCKET}/", "")
                break

        thumbnail_key = None
        if proxy_key:
            print(f"  Generating thumbnail for {video_name}...")
            thumbnail_key = generate_thumbnail(s3_client, S3_BUCKET, proxy_key)
            if thumbnail_key:
                thumbs += 1

        mongodb.store_video_fingerprint(
            video_id=vid,
            visual_fp=fp["visual_fingerprint"],
            audio_fp=fp["audio_fingerprint"],
            transcription_fp=fp["transcription_fingerprint"],
            segment_count=fp["segment_count"],
            total_duration=fp["total_duration"],
            video_name=video_name,
            thumbnail_key=thumbnail_key,
        )
        created += 1
        print(f"  OK {vid} ({video_name}): {fp['segment_count']} segments, {fp['total_duration']:.1f}s, thumb={'yes' if thumbnail_key else 'no'}")

    print(f"\nDone: {created} created, {skipped} skipped, {thumbs} thumbnails generated")

if __name__ == "__main__":
    main()
