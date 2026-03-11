"""
AWS Lambda Handler for Multi-Vector Video Embedding Pipeline

Processes videos from S3 using Bedrock Marengo 3.0 to generate
visual, audio, and transcription embeddings, then stores them
in MongoDB Atlas with HNSW indices.

Test Event Format:
{
    "s3_key": "input/file.mp4",
    "bucket": "your-media-bucket-name"
}
"""

import json
import os
import logging
import hashlib
from typing import Optional

from bedrock_client import BedrockMarengoClient
from mongodb_client import MongoDBEmbeddingClient
from s3_vectors_client import S3VectorsClient
import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def generate_video_id(bucket: str, s3_key: str) -> str:
    """
    Generate a unique video ID from bucket and key.

    Args:
        bucket: S3 bucket name
        s3_key: S3 object key

    Returns:
        SHA256-based unique identifier
    """
    unique_string = f"{bucket}/{s3_key}"
    return hashlib.sha256(unique_string.encode()).hexdigest()[:16]


def lambda_handler(event: dict, context) -> dict:
    """
    Lambda handler for processing video embeddings.

    Args:
        event: Lambda event containing either:
            S3 trigger format:
            - Records[0].s3.bucket.name
            - Records[0].s3.object.key

            Manual invocation format:
            - s3_key: S3 object key for the video
            - bucket: S3 bucket name
            - video_id (optional): Custom video identifier
            - embedding_types (optional): List of embedding types to generate
        context: Lambda context object

    Returns:
        Response dictionary with processing results
    """
    logger.info(f"Processing event: {json.dumps(event)}")

    # Handle S3 trigger event format
    if "Records" in event and event["Records"]:
        record = event["Records"][0]
        if "s3" in record:
            bucket = record["s3"]["bucket"]["name"]
            # URL decode the key (S3 encodes special characters)
            import urllib.parse
            s3_key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
            logger.info(f"S3 trigger event - bucket: {bucket}, key: {s3_key}")
        else:
            s3_key = event.get("s3_key")
            bucket = event.get("bucket")
    else:
        # Manual invocation format
        s3_key = event.get("s3_key")
        bucket = event.get("bucket")

    if not s3_key or not bucket:
        return {
            "statusCode": 400,
            "body": json.dumps({
                "error": "Missing required parameters: 's3_key' and 'bucket' are required",
                "received": {"s3_key": s3_key, "bucket": bucket}
            })
        }

    upload_id = event.get("upload_id")
    status_key = event.get("status_key")

    def update_upload_status(progress, message, status="processing"):
        if status_key:
            try:
                s3_client = boto3.client("s3", region_name="us-east-1")
                s3_client.put_object(
                    Bucket=bucket, Key=status_key,
                    Body=json.dumps({"status": status, "progress": progress, "message": message}),
                    ContentType="application/json"
                )
            except Exception:
                pass  # Best-effort

    # Bedrock doesn't support S3 URIs with spaces - rename file if needed
    original_s3_key = s3_key
    if ' ' in s3_key:
        new_s3_key = s3_key.replace(' ', '-')
        logger.info(f"File has spaces in name, renaming: {s3_key} -> {new_s3_key}")
        s3_client = boto3.client("s3", region_name="us-east-1")
        try:
            # Copy to new name
            copy_source = {"Bucket": bucket, "Key": s3_key}
            s3_client.copy_object(
                CopySource=copy_source,
                Bucket=bucket,
                Key=new_s3_key
            )
            # Delete old name
            s3_client.delete_object(Bucket=bucket, Key=s3_key)
            s3_key = new_s3_key
            logger.info(f"File renamed successfully")
        except Exception as e:
            logger.error(f"Failed to rename file: {str(e)}")
            return {
                "statusCode": 500,
                "body": json.dumps({
                    "error": f"Failed to rename file with spaces: {str(e)}",
                    "original_key": original_s3_key
                })
            }

    # Optional parameters
    video_id = event.get("video_id") or generate_video_id(bucket, s3_key)
    embedding_types = event.get("embedding_types", ["visual", "audio", "transcription"])

    # Segmentation configuration
    segmentation_method = event.get("segmentation_method", "dynamic")
    min_duration_sec = event.get("min_duration_sec", 4)  # For dynamic segmentation
    segment_length_sec = event.get("segment_length_sec", 6)  # For fixed segmentation

    # Storage configuration — only write to selected backends/modes
    storage_backends = event.get("storage_backends", ["mongodb", "s3vectors"])
    index_modes = event.get("index_modes", ["single", "multi"])

    try:
        # Initialize clients
        logger.info("Initializing Bedrock and MongoDB clients...")

        # Output bucket for async results (same as input bucket)
        output_bucket = os.environ.get("OUTPUT_BUCKET", bucket)
        bedrock_client = BedrockMarengoClient(
            region="us-east-1",
            output_bucket=output_bucket,
            output_prefix="embeddings/"
        )

        mongodb_uri = os.environ.get("MONGODB_URI")
        if not mongodb_uri:
            raise ValueError("MONGODB_URI environment variable not set")

        mongodb_client = MongoDBEmbeddingClient(
            connection_string=mongodb_uri,
            database_name=os.environ.get("MONGODB_DATABASE", "video_search")
        )

        # Initialize S3 Vectors client
        s3_vectors_bucket = os.environ.get("S3_VECTORS_BUCKET", "your-vectors-bucket-name")
        s3_vectors_client = S3VectorsClient(
            bucket_name=s3_vectors_bucket,
            region="us-east-1"
        )

        # Initialize S3 client for moving files
        s3_client = boto3.client("s3", region_name="us-east-1")

        # Generate embeddings from video
        logger.info(f"Generating embeddings for s3://{bucket}/{s3_key}")
        logger.info(f"Embedding types: {embedding_types}")
        logger.info(f"Segmentation: {segmentation_method} (minDuration={min_duration_sec}s for dynamic, length={segment_length_sec}s for fixed)")
        logger.info(f"Storage backends: {storage_backends}, Index modes: {index_modes}")

        embeddings_result = bedrock_client.get_video_embeddings(
            bucket=bucket,
            s3_key=s3_key,
            embedding_types=embedding_types,
            segmentation_method=segmentation_method,
            min_duration_sec=min_duration_sec,
            segment_length_sec=segment_length_sec
        )

        segments = embeddings_result.get("segments", [])
        logger.info(f"Generated embeddings for {len(segments)} segments")
        update_upload_status(50, "Segmentation complete. Generating embeddings...")

        if not segments:
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "No segments found in video",
                    "video_id": video_id,
                    "s3_uri": f"s3://{bucket}/{s3_key}"
                })
            }

        # Determine proxy S3 URI (where video will be after moving)
        proxy_key = s3_key
        if s3_key.startswith("input/"):
            proxy_key = s3_key.replace("input/", "proxies/", 1)
        proxy_s3_uri = f"s3://{bucket}/{proxy_key}"

        # Store embeddings in MongoDB (use proxy path)
        logger.info(f"Storing embeddings in MongoDB for video_id: {video_id}")
        logger.info(f"Using proxy S3 URI: {proxy_s3_uri}")

        # Update segments with proxy S3 URI before storing
        for segment in segments:
            segment["s3_uri"] = proxy_s3_uri

        # Store embeddings only in selected backends and index modes
        storage_result = {}
        s3v_result = {}

        if "mongodb" in storage_backends:
            logger.info(f"Storing in MongoDB (index_modes={index_modes})")
            storage_result = mongodb_client.store_all_segments(
                video_id=video_id, segments=segments, index_modes=index_modes
            )
            logger.info(f"MongoDB storage result: {json.dumps(storage_result)}")
        else:
            logger.info("Skipping MongoDB storage (not selected)")

        update_upload_status(70, "Embeddings generated. Storing vectors...")

        if "s3vectors" in storage_backends:
            s3v_dual = "single" in index_modes and "multi" in index_modes
            logger.info(f"Storing in S3 Vectors (dual_write={s3v_dual}, index_modes={index_modes})")
            s3v_result = s3_vectors_client.store_all_segments(video_id, segments, dual_write=s3v_dual)
            logger.info(f"S3 Vectors storage result: {json.dumps(s3v_result)}")
        else:
            logger.info("Skipping S3 Vectors storage (not selected)")

        update_upload_status(85, "Vectors stored. Generating thumbnail...")

        # Move video from input/ to proxies/ if needed
        moved = False
        if s3_key.startswith("input/"):
            logger.info(f"Moving video from input/ to proxies/")
            try:
                # Derive correct content type from extension (browser needs video/mp4 to play)
                ext = os.path.splitext(proxy_key)[1].lower().lstrip(".")
                content_type_map = {
                    "mp4": "video/mp4", "mov": "video/quicktime",
                    "mxf": "application/mxf", "avi": "video/x-msvideo",
                    "mkv": "video/x-matroska", "webm": "video/webm",
                }
                proxy_content_type = content_type_map.get(ext, "application/octet-stream")

                # Copy to proxy location with correct ContentType
                copy_source = {"Bucket": bucket, "Key": s3_key}
                s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=bucket,
                    Key=proxy_key,
                    ServerSideEncryption='AES256',
                    ContentType=proxy_content_type,
                    MetadataDirective='REPLACE'
                )

                # Delete from Ready location
                s3_client.delete_object(Bucket=bucket, Key=s3_key)
                logger.info(f"Successfully moved video to {proxy_key}")
                moved = True
            except Exception as e:
                logger.error(f"Failed to move video: {str(e)}")
                # Don't fail the whole Lambda if move fails
        else:
            logger.info("Video not in input/ folder, skipping move")

        # Transcode proxy to web-friendly format + generate thumbnail
        import subprocess
        import tempfile

        thumbnail_key = proxy_key.rsplit(".", 1)[0] + "_thumb.jpg" if proxy_key else None
        tmp_video_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
                s3_client.download_file(bucket, proxy_key, tmp_video.name)
                tmp_video_path = tmp_video.name

            # Check if transcoding is needed (bitrate > 5 Mbps)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=bit_rate,width,height",
                 "-of", "csv=p=0", tmp_video_path],
                capture_output=True, text=True, timeout=15
            )
            needs_transcode = False
            if probe.returncode == 0 and probe.stdout.strip():
                parts = probe.stdout.strip().split(",")
                try:
                    vid_width = int(parts[0])
                    vid_height = int(parts[1])
                    vid_bitrate = int(parts[2]) if len(parts) > 2 and parts[2] != "N/A" else 0
                    if vid_bitrate > 5_000_000 or vid_width > 1920:
                        needs_transcode = True
                        logger.info(f"Video needs transcoding: {vid_width}x{vid_height} @ {vid_bitrate/1e6:.1f} Mbps")
                except (ValueError, IndexError):
                    pass

            if needs_transcode:
                import time
                import threading

                # Get video duration for progress estimation
                dur_probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", tmp_video_path],
                    capture_output=True, text=True, timeout=15
                )
                total_duration_sec = 0
                try:
                    total_duration_sec = float(dur_probe.stdout.strip())
                except (ValueError, AttributeError):
                    pass

                update_upload_status(86, f"Transcoding to 720p for web playback (0%)...")
                tmp_transcoded = tmp_video_path.rsplit(".", 1)[0] + "_web.mp4"
                progress_file = tmp_video_path.rsplit(".", 1)[0] + "_progress.log"

                # Run ffmpeg with progress output
                tc_proc = subprocess.Popen([
                    "ffmpeg", "-y", "-i", tmp_video_path,
                    "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-maxrate", "4M", "-bufsize", "8M",
                    "-vf", "scale='min(1280,iw)':'min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    "-progress", progress_file, "-nostats",
                    tmp_transcoded
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Poll progress file while ffmpeg runs
                last_pct = 0
                while tc_proc.poll() is None:
                    time.sleep(3)
                    if total_duration_sec > 0 and os.path.exists(progress_file):
                        try:
                            with open(progress_file, "r") as pf:
                                content = pf.read()
                            # Parse out_time_us (microseconds of progress)
                            lines = content.strip().split("\n")
                            for line in reversed(lines):
                                if line.startswith("out_time_us="):
                                    us = int(line.split("=")[1])
                                    tc_pct = min(int((us / 1e6) / total_duration_sec * 100), 99)
                                    if tc_pct > last_pct:
                                        last_pct = tc_pct
                                        # Map transcode 0-100% to upload progress 86-96%
                                        upload_pct = 86 + int(tc_pct * 0.10)
                                        update_upload_status(upload_pct, f"Transcoding to 720p for web playback ({tc_pct}%)...")
                                    break
                        except Exception:
                            pass

                tc_proc.wait()
                # Cleanup progress file
                if os.path.exists(progress_file):
                    os.unlink(progress_file)

                if tc_proc.returncode == 0 and os.path.exists(tmp_transcoded) and os.path.getsize(tmp_transcoded) > 0:
                    update_upload_status(96, "Uploading transcoded proxy...")
                    ext = os.path.splitext(proxy_key)[1].lower().lstrip(".")
                    content_type_map = {
                        "mp4": "video/mp4", "mov": "video/quicktime",
                        "mxf": "application/mxf",
                    }
                    s3_client.upload_file(
                        tmp_transcoded, bucket, proxy_key,
                        ExtraArgs={
                            "ContentType": content_type_map.get(ext, "video/mp4"),
                            "ServerSideEncryption": "AES256",
                        }
                    )
                    old_size = os.path.getsize(tmp_video_path)
                    new_size = os.path.getsize(tmp_transcoded)
                    logger.info(f"Transcoded proxy: {old_size/1e6:.0f}MB -> {new_size/1e6:.0f}MB")
                    update_upload_status(97, f"Transcoded: {old_size/1e6:.0f}MB to {new_size/1e6:.0f}MB")
                    # Use transcoded file for thumbnail
                    os.unlink(tmp_video_path)
                    tmp_video_path = tmp_transcoded
                else:
                    stderr_out = tc_proc.stderr.read().decode() if tc_proc.stderr else ""
                    logger.warning(f"Transcode failed (non-fatal): {stderr_out[:500]}")
                    if os.path.exists(tmp_transcoded):
                        os.unlink(tmp_transcoded)
            else:
                update_upload_status(86, "Video is web-ready, skipping transcode.")

            # Generate thumbnail from (possibly transcoded) video
            tmp_thumb_path = tmp_video_path.rsplit(".", 1)[0] + "_thumb.jpg"
            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_video_path,
                "-ss", "2", "-vframes", "1",
                "-vf", "scale=480:-1",
                "-q:v", "3",
                tmp_thumb_path
            ], capture_output=True, timeout=30)

            if os.path.exists(tmp_thumb_path) and os.path.getsize(tmp_thumb_path) > 0:
                s3_client.upload_file(
                    tmp_thumb_path, bucket, thumbnail_key,
                    ExtraArgs={"ContentType": "image/jpeg"}
                )
                logger.info(f"Generated thumbnail: {thumbnail_key}")
            else:
                logger.warning("Thumbnail generation produced empty file")
                thumbnail_key = None

            # Cleanup temp files
            if tmp_video_path and os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)
            if os.path.exists(tmp_thumb_path):
                os.unlink(tmp_thumb_path)
        except Exception as e:
            logger.warning(f"Transcode/thumbnail failed (non-fatal): {e}")
            thumbnail_key = None
            if tmp_video_path and os.path.exists(tmp_video_path):
                try: os.unlink(tmp_video_path)
                except: pass

        # Compute and store video fingerprint
        try:
            from compare_client import compute_fingerprint
            fp = compute_fingerprint(segments)
            video_name = s3_key.split("/")[-1] if s3_key else video_id
            mongodb_client.store_video_fingerprint(
                video_id=video_id,
                visual_fp=fp["visual_fingerprint"],
                audio_fp=fp["audio_fingerprint"],
                transcription_fp=fp["transcription_fingerprint"],
                segment_count=fp["segment_count"],
                total_duration=fp["total_duration"],
                video_name=video_name,
                thumbnail_key=thumbnail_key,
            )
            logger.info(f"Stored fingerprint for video {video_id}")
        except Exception as e:
            logger.warning(f"Failed to store fingerprint for {video_id}: {e}")

        update_upload_status(100, "Processing complete.", status="complete")

        # Close MongoDB connection
        mongodb_client.close()

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Video processed successfully",
                "video_id": video_id,
                "original_s3_uri": f"s3://{bucket}/{s3_key}",
                "proxy_s3_uri": proxy_s3_uri,
                "video_moved": moved,
                "segments_processed": storage_result.get("segments_processed", 0),
                "embeddings_stored": {
                    "mongodb": {
                        "visual": storage_result.get("visual_stored", 0),
                        "audio": storage_result.get("audio_stored", 0),
                        "transcription": storage_result.get("transcription_stored", 0)
                    },
                    "s3_vectors": {
                        "visual": s3v_result.get("visual_stored", 0),
                        "audio": s3v_result.get("audio_stored", 0),
                        "transcription": s3v_result.get("transcription_stored", 0)
                    }
                },
                "metadata": embeddings_result.get("metadata", {})
            })
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        update_upload_status(0, f"Error: {str(e)}", status="error")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "video_id": video_id,
                "s3_uri": f"s3://{bucket}/{s3_key}"
            })
        }


# For local testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Test event
    test_event = {
        "s3_key": "input/test.mp4",
        "bucket": "your-media-bucket-name"
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(json.loads(result["body"]), indent=2))
