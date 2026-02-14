"""
Video Subclip Lambda — creates subclips and concatenates segments using FFmpeg.

Supports two operations:
1. create_subclip: Extract a time range from a video
2. concatenate: Merge multiple subclips into one video

Input videos are read from S3, processed with FFmpeg, and output is
written to s3://{S3_BUCKET}/temp/subclips/{uuid}.mp4

Environment:
    S3_BUCKET: S3 bucket for input videos and output subclips
    FFMPEG_PATH: Path to FFmpeg binary (default: /opt/bin/ffmpeg from layer)
"""

import json
import os
import subprocess
import uuid
import boto3

s3 = boto3.client("s3")

S3_BUCKET = os.environ.get("S3_BUCKET", "multi-modal-video-search-app")
FFMPEG = os.environ.get("FFMPEG_PATH", "/opt/bin/ffmpeg")
TMP_DIR = "/tmp"


def lambda_handler(event, context):
    """
    Handle subclip operations.

    Event format:
    {
        "operation": "create_subclip" | "concatenate",

        # For create_subclip:
        "s3_uri": "s3://bucket/key",
        "start_time": 150.0,
        "end_time": 156.0,

        # For concatenate:
        "clips": [
            {"s3_uri": "s3://bucket/key1", "start_time": 10, "end_time": 16},
            {"s3_uri": "s3://bucket/key2", "start_time": 30, "end_time": 36}
        ]
    }

    Returns:
    {
        "s3_uri": "s3://bucket/temp/subclips/uuid.mp4",
        "duration": 6.0
    }
    """
    operation = event.get("operation", "create_subclip")

    if operation == "create_subclip":
        return handle_create_subclip(event)
    elif operation == "concatenate":
        return handle_concatenate(event)
    else:
        return {"error": f"Unknown operation: {operation}"}


def handle_create_subclip(event):
    """Extract a time range from a video."""
    s3_uri = event["s3_uri"]
    start_time = float(event["start_time"])
    end_time = float(event["end_time"])
    duration = end_time - start_time

    if duration <= 0:
        return {"error": "end_time must be greater than start_time"}
    if duration > 3600:
        return {"error": "Subclip duration exceeds 1 hour limit"}

    # Download source video
    bucket, key = parse_s3_uri(s3_uri)
    input_path = os.path.join(TMP_DIR, f"input_{uuid.uuid4().hex[:8]}.mp4")
    print(f"Downloading s3://{bucket}/{key} to {input_path}")
    s3.download_file(bucket, key, input_path)

    # Create subclip with FFmpeg
    output_id = uuid.uuid4().hex[:12]
    output_path = os.path.join(TMP_DIR, f"subclip_{output_id}.mp4")

    cmd = [
        FFMPEG, "-y",
        "-ss", str(start_time),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",
        "-movflags", "+faststart",
        output_path
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"FFmpeg stderr: {result.stderr}")
        return {"error": f"FFmpeg failed: {result.stderr[-500:]}"}

    # Upload to S3
    output_key = f"temp/subclips/{output_id}.mp4"
    print(f"Uploading to s3://{S3_BUCKET}/{output_key}")
    s3.upload_file(output_path, S3_BUCKET, output_key)

    # Cleanup
    cleanup(input_path, output_path)

    return {
        "s3_uri": f"s3://{S3_BUCKET}/{output_key}",
        "duration": duration
    }


def handle_concatenate(event):
    """Create subclips then concatenate them."""
    clips = event.get("clips", [])
    if not clips:
        return {"error": "No clips provided"}
    if len(clips) > 10:
        return {"error": "Maximum 10 clips for concatenation"}

    subclip_paths = []
    temp_files = []
    downloaded = {}  # s3_uri -> local path (reuse for same-video clips)

    try:
        # Create each subclip
        for i, clip in enumerate(clips):
            s3_uri = clip["s3_uri"]
            start_time = float(clip["start_time"])
            end_time = float(clip["end_time"])
            duration = end_time - start_time

            # Reuse already-downloaded source video (highlight reels = same video)
            if s3_uri in downloaded:
                input_path = downloaded[s3_uri]
                print(f"Clip {i}: reusing cached {input_path}")
            else:
                bucket, key = parse_s3_uri(s3_uri)
                input_path = os.path.join(TMP_DIR, f"concat_input_{len(downloaded)}.mp4")
                print(f"Clip {i}: downloading s3://{bucket}/{key} to {input_path}")
                s3.download_file(bucket, key, input_path)
                downloaded[s3_uri] = input_path
                temp_files.append(input_path)

            subclip_path = os.path.join(TMP_DIR, f"concat_sub_{i}.mp4")
            # Re-encode with fade in/out for smooth transitions between clips
            fade_dur = 0.4
            fade_out_start = max(duration - fade_dur, 0.1)
            cmd = [
                FFMPEG, "-y",
                "-ss", str(start_time),
                "-i", input_path,
                "-t", str(duration),
                "-vf", f"fade=in:0:d={fade_dur},fade=out:st={fade_out_start}:d={fade_dur}",
                "-af", f"afade=in:0:d={fade_dur},afade=out:st={fade_out_start}:d={fade_dur}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                subclip_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                return {"error": f"FFmpeg subclip {i} failed: {result.stderr[-500:]}"}
            subclip_paths.append(subclip_path)
            temp_files.append(subclip_path)

        # Write concat list file
        list_path = os.path.join(TMP_DIR, "concat_list.txt")
        with open(list_path, "w") as f:
            for p in subclip_paths:
                f.write(f"file '{p}'\n")
        temp_files.append(list_path)

        # Concatenate
        output_id = uuid.uuid4().hex[:12]
        output_path = os.path.join(TMP_DIR, f"concat_{output_id}.mp4")

        cmd = [
            FFMPEG, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path
        ]
        print(f"Concatenating {len(subclip_paths)} clips")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {"error": f"FFmpeg concat failed: {result.stderr[-500:]}"}

        # Upload
        output_key = f"temp/subclips/{output_id}.mp4"
        s3.upload_file(output_path, S3_BUCKET, output_key)
        temp_files.append(output_path)

        total_duration = sum(float(c["end_time"]) - float(c["start_time"]) for c in clips)

        return {
            "s3_uri": f"s3://{S3_BUCKET}/{output_key}",
            "duration": total_duration,
            "clip_count": len(clips)
        }
    finally:
        for f in temp_files:
            cleanup(f)


def parse_s3_uri(s3_uri: str) -> tuple:
    """Parse s3://bucket/key into (bucket, key)."""
    if s3_uri.startswith("s3://"):
        parts = s3_uri[5:].split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""
    raise ValueError(f"Invalid S3 URI: {s3_uri}")


def cleanup(*paths):
    """Remove temp files."""
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
