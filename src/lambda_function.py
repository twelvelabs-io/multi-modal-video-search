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
import re
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

# Cached path for ffmpeg binary (downloaded from S3 on first use)
_ffmpeg_path = None
TOOLS_BUCKET = "multi-modal-video-search-app"


def _ensure_ffmpeg():
    """Download ffmpeg from S3 to /tmp if not already present."""
    global _ffmpeg_path
    if _ffmpeg_path and os.path.exists(_ffmpeg_path):
        return _ffmpeg_path

    local = "/tmp/ffmpeg"
    if not os.path.exists(local):
        s3 = boto3.client("s3", region_name="us-east-1")
        logger.info(f"Downloading ffmpeg from s3://{TOOLS_BUCKET}/tools/ffmpeg")
        s3.download_file(TOOLS_BUCKET, "tools/ffmpeg", local)
        os.chmod(local, 0o755)
    else:
        logger.info("ffmpeg already cached at /tmp/ffmpeg")

    _ffmpeg_path = local
    return _ffmpeg_path


def _parse_technical_metadata(probe_output: str, filename: str = "", s3_uri: str = "", file_size_bytes: int = 0) -> dict:
    """Parse ffmpeg -i stderr output into comprehensive technical metadata."""
    metadata = {
        "container": {},
        "video": {},
        "audio": {},
        "source": {"filename": filename, "s3_uri": s3_uri},
    }

    # Container format: "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '...'"
    fmt_match = re.search(r'Input #\d+,\s*([^,]+(?:,[^,]+)*),\s*from', probe_output)
    if fmt_match:
        metadata["container"]["format"] = fmt_match.group(1).strip()

    # Duration + overall bitrate: "Duration: 00:04:51.00, start: 0.000, bitrate: 10234 kb/s"
    dur_match = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', probe_output)
    if dur_match:
        dur_sec = int(dur_match.group(1)) * 3600 + int(dur_match.group(2)) * 60 + float(dur_match.group(3))
        metadata["container"]["duration"] = round(dur_sec, 2)

    br_match = re.search(r'bitrate:\s*(\d+)\s*kb/s', probe_output)
    if br_match:
        metadata["container"]["bitrate_kbps"] = int(br_match.group(1))

    if file_size_bytes > 0:
        metadata["container"]["file_size_mb"] = round(file_size_bytes / (1024 * 1024), 2)

    # Video stream: "Stream #0:0...: Video: h264 (High), yuv420p(tv, bt709), 1920x1080 [SAR 1:1 DAR 16:9], 25 fps, ..."
    video_match = re.search(r'Stream #\d+:\d+[^:]*:\s*Video:\s*(.+)', probe_output)
    if video_match:
        vline = video_match.group(1)

        # Codec + profile: "h264 (High)" or "hevc (Main 10)"
        codec_match = re.match(r'(\w+)(?:\s*\(([^)]+)\))?', vline)
        if codec_match:
            metadata["video"]["codec"] = codec_match.group(1)
            if codec_match.group(2):
                profile_str = codec_match.group(2).strip()
                # "High 4:4:4 Predictive" → profile="High 4:4:4 Predictive"
                # "High" → profile="High"
                metadata["video"]["profile"] = profile_str

        # Color space: "yuv420p" or "yuv420p10le" etc.
        cs_match = re.search(r',\s*(yuv\w+|rgb\w+|gbr\w+|gray\w*|nv\d+)', vline)
        if cs_match:
            color_space = cs_match.group(1)
            metadata["video"]["color_space"] = color_space
            # Bit depth from pixel format
            if "10le" in color_space or "10be" in color_space:
                metadata["video"]["bit_depth"] = 10
            elif "12le" in color_space or "12be" in color_space:
                metadata["video"]["bit_depth"] = 12
            else:
                metadata["video"]["bit_depth"] = 8

        # Resolution: "1920x1080"
        res_match = re.search(r'(\d{2,5})x(\d{2,5})', vline)
        if res_match:
            metadata["video"]["width"] = int(res_match.group(1))
            metadata["video"]["height"] = int(res_match.group(2))

        # DAR: "DAR 16:9"
        dar_match = re.search(r'DAR\s+(\d+:\d+)', vline)
        if dar_match:
            metadata["video"]["display_aspect_ratio"] = dar_match.group(1)

        # Framerate: "25 fps" or "29.97 fps" or "25 tbr"
        fps_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:fps|tbr)', vline)
        if fps_match:
            metadata["video"]["framerate"] = float(fps_match.group(1))

        # Video stream bitrate: "10000 kb/s" (at end of stream line)
        vbr_match = re.search(r'(\d+)\s*kb/s', vline)
        if vbr_match:
            metadata["video"]["bitrate_kbps"] = int(vbr_match.group(1))

        # Scan type — interlaced if "tff" or "bff" in stream
        if re.search(r'\b(tff|bff|interlaced)\b', vline, re.IGNORECASE):
            metadata["video"]["scan_type"] = "interlaced"
        else:
            metadata["video"]["scan_type"] = "progressive"

    # Audio stream: "Stream #0:1...: Audio: aac (LC), 48000 Hz, stereo, fltp, 128 kb/s"
    audio_match = re.search(r'Stream #\d+:\d+[^:]*:\s*Audio:\s*(.+)', probe_output)
    if audio_match:
        aline = audio_match.group(1)

        # Codec: "aac" or "aac (LC)" or "pcm_s24le"
        acodec_match = re.match(r'(\w+)', aline)
        if acodec_match:
            metadata["audio"]["codec"] = acodec_match.group(1)

        # Sample rate: "48000 Hz"
        sr_match = re.search(r'(\d+)\s*Hz', aline)
        if sr_match:
            metadata["audio"]["sample_rate"] = int(sr_match.group(1))

        # Channel layout: "stereo", "5.1", "mono", "5.1(side)"
        ch_match = re.search(r'Hz,\s*(\w+(?:\.\d+)?(?:\([^)]*\))?)', aline)
        if ch_match:
            layout = ch_match.group(1)
            metadata["audio"]["channel_layout"] = layout
            ch_count_map = {"mono": 1, "stereo": 2, "5.1": 6, "5.1(side)": 6, "7.1": 8}
            metadata["audio"]["channels"] = ch_count_map.get(layout.split("(")[0], 0)

        # Audio bitrate: "128 kb/s"
        abr_match = re.search(r'(\d+)\s*kb/s', aline)
        if abr_match:
            metadata["audio"]["bitrate_kbps"] = int(abr_match.group(1))

    return metadata


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
    segmentation_method = event.get("segmentation_method", "fixed")
    min_duration_sec = event.get("min_duration_sec", 4)  # For dynamic segmentation
    segment_length_sec = event.get("segment_length_sec", 1)  # For fixed segmentation (Marengo range: 1-10)

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

        update_upload_status(55, "Moving video to proxy location...")

        # Move video from input/ to originals/ + proxies/
        moved = False
        if s3_key.startswith("input/"):
            logger.info(f"Moving video from input/ to originals/ and proxies/")
            try:
                # Derive correct content type from extension (browser needs video/mp4 to play)
                ext = os.path.splitext(proxy_key)[1].lower().lstrip(".")
                content_type_map = {
                    "mp4": "video/mp4", "mov": "video/quicktime",
                    "mxf": "application/mxf", "avi": "video/x-msvideo",
                    "mkv": "video/x-matroska", "webm": "video/webm",
                }
                proxy_content_type = content_type_map.get(ext, "application/octet-stream")
                copy_source = {"Bucket": bucket, "Key": s3_key}

                # Preserve original high-res in originals/
                original_key = s3_key.replace("input/", "originals/", 1)
                s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=bucket,
                    Key=original_key,
                    ServerSideEncryption='AES256',
                    MetadataDirective='COPY'
                )
                logger.info(f"Preserved original at {original_key}")

                # Copy to proxy location with correct ContentType
                s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=bucket,
                    Key=proxy_key,
                    ServerSideEncryption='AES256',
                    ContentType=proxy_content_type,
                    MetadataDirective='REPLACE'
                )

                # Delete from input/
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

        # Ensure ffmpeg is available (download from S3 on cold start)
        ffmpeg_bin = _ensure_ffmpeg()

        thumbnail_key = proxy_key.rsplit(".", 1)[0] + "_thumb.jpg" if proxy_key else None
        tmp_video_path = None
        technical_metadata = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
                s3_client.download_file(bucket, proxy_key, tmp_video.name)
                tmp_video_path = tmp_video.name

            # Parse full technical metadata using ffmpeg -i stderr
            probe = subprocess.run(
                [ffmpeg_bin, "-i", tmp_video_path, "-hide_banner"],
                capture_output=True, text=True, timeout=15
            )
            probe_output = probe.stderr or ""

            # Get file size for metadata
            file_size_bytes = os.path.getsize(tmp_video_path)
            source_filename = s3_key.split("/")[-1] if s3_key else ""
            original_s3_uri = f"s3://{bucket}/{s3_key.replace('input/', 'originals/', 1)}" if s3_key.startswith("input/") else f"s3://{bucket}/{s3_key}"

            technical_metadata = _parse_technical_metadata(
                probe_output, filename=source_filename,
                s3_uri=original_s3_uri, file_size_bytes=file_size_bytes
            )
            logger.info(f"Technical metadata: {json.dumps(technical_metadata)}")

            # Always transcode to produce a proper web proxy (720p, faststart, reasonable size)
            # Embeddings are generated from the source; proxy is only for browser playback
            needs_transcode = True
            vid_width = technical_metadata["video"].get("width", 0)
            vid_height = technical_metadata["video"].get("height", 0)
            vid_bitrate_kbps = technical_metadata["container"].get("bitrate_kbps", 0)
            total_duration_sec = technical_metadata["container"].get("duration", 0)
            logger.info(f"Transcoding to web proxy: {vid_width}x{vid_height} @ {vid_bitrate_kbps/1000:.1f} Mbps")

            if needs_transcode:
                import time

                update_upload_status(60, "Transcoding to 720p via MediaConvert...")

                mc_endpoint = os.environ.get("MEDIACONVERT_ENDPOINT", "")
                mc_role_arn = os.environ.get("MEDIACONVERT_ROLE_ARN", "")

                if mc_endpoint and mc_role_arn:
                    # Use AWS MediaConvert (hardware-accelerated, handles any file size)
                    mc_client = boto3.client("mediaconvert", region_name="us-east-1", endpoint_url=mc_endpoint)

                    input_s3 = f"s3://{bucket}/{proxy_key}"
                    # Output to a temp prefix — MediaConvert can't overwrite input
                    output_prefix = f"s3://{bucket}/transcode-tmp/"
                    base_name = os.path.splitext(os.path.basename(proxy_key))[0]

                    job_settings = {
                        "Inputs": [{
                            "FileInput": input_s3,
                            "AudioSelectors": {"Audio Selector 1": {"DefaultSelection": "DEFAULT"}},
                            "VideoSelector": {}
                        }],
                        "OutputGroups": [{
                            "Name": "File Group",
                            "OutputGroupSettings": {
                                "Type": "FILE_GROUP_SETTINGS",
                                "FileGroupSettings": {"Destination": output_prefix}
                            },
                            "Outputs": [{
                                "NameModifier": "",
                                "ContainerSettings": {
                                    "Container": "MP4",
                                    "Mp4Settings": {"MoovPlacement": "PROGRESSIVE_DOWNLOAD"}
                                },
                                "VideoDescription": {
                                    "CodecSettings": {
                                        "Codec": "H_264",
                                        "H264Settings": {
                                            "RateControlMode": "QVBR",
                                            "QvbrSettings": {"QvbrQualityLevel": 7},
                                            "MaxBitrate": 4000000,
                                            "CodecProfile": "MAIN",
                                            "CodecLevel": "AUTO",
                                        }
                                    },
                                    "Width": min(1280, vid_width),
                                    "Height": min(720, vid_height),
                                    "ScalingBehavior": "DEFAULT",
                                    "AntiAlias": "ENABLED",
                                },
                                "AudioDescriptions": [{
                                    "CodecSettings": {
                                        "Codec": "AAC",
                                        "AacSettings": {
                                            "Bitrate": 128000,
                                            "CodingMode": "CODING_MODE_2_0",
                                            "SampleRate": 48000
                                        }
                                    },
                                    "AudioSourceName": "Audio Selector 1"
                                }]
                            }]
                        }]
                    }

                    job = mc_client.create_job(Role=mc_role_arn, Settings=job_settings)
                    job_id = job["Job"]["Id"]
                    logger.info(f"MediaConvert job submitted: {job_id}")

                    # Poll until complete
                    mc_status = "SUBMITTED"
                    while mc_status in ("SUBMITTED", "PROGRESSING"):
                        time.sleep(10)
                        status_resp = mc_client.get_job(Id=job_id)
                        mc_status = status_resp["Job"]["Status"]
                        pct = status_resp["Job"].get("JobPercentComplete", 0)
                        upload_pct = 60 + int(pct * 0.30)
                        update_upload_status(upload_pct, f"Transcoding via MediaConvert ({pct}%)...")

                    if mc_status == "COMPLETE":
                        logger.info(f"MediaConvert job complete: {job_id}")
                        # Move transcoded file from temp prefix to proxy location
                        tc_key = f"transcode-tmp/{base_name}.mp4"

                        # Ensure proxy key has .mp4 extension (source may be .mxf, .avi, etc.)
                        old_proxy_key = proxy_key
                        if not proxy_key.lower().endswith(".mp4"):
                            proxy_key = os.path.splitext(proxy_key)[0] + ".mp4"
                            proxy_s3_uri = f"s3://{bucket}/{proxy_key}"
                            logger.info(f"Renamed proxy key: {old_proxy_key} -> {proxy_key}")

                        s3_client.copy_object(
                            Bucket=bucket,
                            CopySource={"Bucket": bucket, "Key": tc_key},
                            Key=proxy_key,
                            ContentType="video/mp4",
                            ServerSideEncryption="AES256",
                            MetadataDirective="REPLACE"
                        )
                        s3_client.delete_object(Bucket=bucket, Key=tc_key)
                        # Delete old proxy with non-mp4 extension if it was renamed
                        if old_proxy_key != proxy_key:
                            try:
                                s3_client.delete_object(Bucket=bucket, Key=old_proxy_key)
                                logger.info(f"Deleted old proxy: {old_proxy_key}")
                            except Exception:
                                pass

                        # Update thumbnail key to match new proxy key
                        thumbnail_key = proxy_key.rsplit(".", 1)[0] + "_thumb.jpg"

                        update_upload_status(91, "Transcode complete.")
                        # Re-download transcoded proxy for thumbnail
                        os.unlink(tmp_video_path)
                        s3_client.download_file(bucket, proxy_key, tmp_video_path)
                    else:
                        error_msg = status_resp["Job"].get("ErrorMessage", "Unknown")
                        logger.warning(f"MediaConvert failed (non-fatal): {error_msg}")
                else:
                    logger.warning("MediaConvert not configured, skipping transcode")
                    update_upload_status(90, "Transcode skipped (MediaConvert not configured)")
            else:
                update_upload_status(90, "Video is web-ready, skipping transcode.")

            # Generate thumbnail from (possibly transcoded) video
            tmp_thumb_path = tmp_video_path.rsplit(".", 1)[0] + "_thumb.jpg"
            subprocess.run([
                ffmpeg_bin, "-y", "-i", tmp_video_path,
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

        # Store embeddings AFTER proxy is ready (transcoded + thumbnail done)
        update_upload_status(95, "Storing embeddings in vector indexes...")
        logger.info(f"Storing embeddings in MongoDB for video_id: {video_id}")
        logger.info(f"Using proxy S3 URI: {proxy_s3_uri}")

        # Update segments with proxy S3 URI before storing
        for segment in segments:
            segment["s3_uri"] = proxy_s3_uri

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

        if "s3vectors" in storage_backends:
            s3v_dual = "single" in index_modes and "multi" in index_modes
            logger.info(f"Storing in S3 Vectors (dual_write={s3v_dual}, index_modes={index_modes})")
            s3v_result = s3_vectors_client.store_all_segments(video_id, segments, dual_write=s3v_dual)
            logger.info(f"S3 Vectors storage result: {json.dumps(s3v_result)}")
        else:
            logger.info("Skipping S3 Vectors storage (not selected)")

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
                technical_metadata=technical_metadata,
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
