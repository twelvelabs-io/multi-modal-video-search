"""
AWS Lambda Handler for Multi-Vector Video Embedding Pipeline

Processes videos from S3 using Bedrock Marengo 3.0 to generate
visual, audio, and transcription embeddings, then stores them
in MongoDB Atlas with HNSW indices.

Test Event Format:
{
    "s3_key": "WBD_project/Videos/file.mp4",
    "bucket": "tl-brice-media"
}
"""

import json
import os
import logging
import hashlib
from typing import Optional

from bedrock_client import BedrockMarengoClient
from mongodb_client import MongoDBEmbeddingClient

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

    # Optional parameters
    video_id = event.get("video_id") or generate_video_id(bucket, s3_key)
    embedding_types = event.get("embedding_types", ["visual", "audio", "transcription"])

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

        # Generate embeddings from video
        logger.info(f"Generating embeddings for s3://{bucket}/{s3_key}")
        logger.info(f"Embedding types: {embedding_types}")

        embeddings_result = bedrock_client.get_video_embeddings(
            bucket=bucket,
            s3_key=s3_key,
            embedding_types=embedding_types
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

        # Store embeddings in MongoDB
        logger.info(f"Storing embeddings in MongoDB for video_id: {video_id}")

        storage_result = mongodb_client.store_all_segments(
            video_id=video_id,
            segments=segments
        )

        logger.info(f"Storage result: {json.dumps(storage_result)}")

        # Close MongoDB connection
        mongodb_client.close()

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Video processed successfully",
                "video_id": video_id,
                "s3_uri": f"s3://{bucket}/{s3_key}",
                "segments_processed": storage_result["segments_processed"],
                "embeddings_stored": {
                    "visual": storage_result["visual_stored"],
                    "audio": storage_result["audio_stored"],
                    "transcription": storage_result["transcription_stored"]
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
        "bucket": "tl-brice-media"
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(json.loads(result["body"]), indent=2))
