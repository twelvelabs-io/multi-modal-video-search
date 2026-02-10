"""
AWS Lambda Handler for Multi-Vector Video Embedding Pipeline

Processes videos from S3 using Bedrock Marengo 3.0 to generate
visual, audio, and transcription embeddings, then stores them
in MongoDB Atlas with HNSW indices.

Test Event Format:
{
    "s3_key": "WBD_project/Videos/file.mp4",
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
        if "WBD_project/Videos/Ready/" in s3_key:
            proxy_key = s3_key.replace("WBD_project/Videos/Ready/", "WBD_project/Videos/proxy/")
        proxy_s3_uri = f"s3://{bucket}/{proxy_key}"

        # Store embeddings in MongoDB (use proxy path)
        logger.info(f"Storing embeddings in MongoDB for video_id: {video_id}")
        logger.info(f"Using proxy S3 URI: {proxy_s3_uri}")

        # Update segments with proxy S3 URI before storing
        for segment in segments:
            segment["s3_uri"] = proxy_s3_uri

        # Store in both unified and modality-specific collections (dual-write mode)
        storage_result = mongodb_client.store_all_segments(
            video_id=video_id,
            segments=segments,
            dual_write=True
        )

        logger.info(f"MongoDB storage result: {json.dumps(storage_result)}")

        # Store embeddings in S3 Vectors (dual-write to both unified and multi-index)
        logger.info(f"Storing embeddings in S3 Vectors for video_id: {video_id}")
        s3v_result = s3_vectors_client.store_all_segments(video_id, segments, dual_write=True)
        logger.info(f"S3 Vectors storage result: {json.dumps(s3v_result)}")

        # Move video from Ready/ to proxy/ if needed
        moved = False
        if "WBD_project/Videos/Ready/" in s3_key:
            logger.info(f"Moving video from Ready/ to proxy/")
            try:
                # Copy to proxy location
                copy_source = {"Bucket": bucket, "Key": s3_key}
                s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=bucket,
                    Key=proxy_key,
                    ServerSideEncryption='AES256',
                    TaggingDirective='COPY',
                    MetadataDirective='COPY'
                )

                # Delete from Ready location
                s3_client.delete_object(Bucket=bucket, Key=s3_key)
                logger.info(f"Successfully moved video to {proxy_key}")
                moved = True
            except Exception as e:
                logger.error(f"Failed to move video: {str(e)}")
                # Don't fail the whole Lambda if move fails
        else:
            logger.info("Video not in Ready folder, skipping move")

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
                "segments_processed": storage_result["segments_processed"],
                "embeddings_stored": {
                    "mongodb": {
                        "visual": storage_result["visual_stored"],
                        "audio": storage_result["audio_stored"],
                        "transcription": storage_result["transcription_stored"]
                    },
                    "s3_vectors": {
                        "visual": s3v_result["visual_stored"],
                        "audio": s3v_result["audio_stored"],
                        "transcription": s3v_result["transcription_stored"]
                    }
                },
                "metadata": embeddings_result.get("metadata", {})
            })
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
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
        "s3_key": "WBD_project/Videos/test.mp4",
        "bucket": "your-media-bucket-name"
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(json.loads(result["body"]), indent=2))
