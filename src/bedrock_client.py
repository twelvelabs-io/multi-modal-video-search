"""
Bedrock Marengo Client for Multi-Vector Video Embeddings

Calls AWS Bedrock's TwelveLabs Marengo 3.0 model to generate
visual, audio, and transcription embeddings from video content.

Video processing uses async invocation (StartAsyncInvoke).
Text/image queries use sync invocation (InvokeModel).
"""

import json
import os
import time
from typing import Optional
import boto3
from botocore.config import Config


class BedrockMarengoClient:
    """Client for TwelveLabs Marengo 3.0 on AWS Bedrock."""

    MODEL_ID = "twelvelabs.marengo-embed-3-0-v1:0"
    EMBEDDING_DIMENSION = 512

    # Async invocation polling settings
    POLL_INTERVAL_SECONDS = 10
    MAX_POLL_ATTEMPTS = 360  # 1 hour max wait

    def __init__(
        self,
        region: str = "us-east-1",
        output_bucket: Optional[str] = None,
        output_prefix: str = "marengo-output/",
        account_id: Optional[str] = None,
        s3_client: Optional[boto3.client] = None,
        bedrock_client: Optional[boto3.client] = None
    ):
        """
        Initialize the Bedrock Marengo client.

        Args:
            region: AWS region (must be us-east-1 for Marengo)
            output_bucket: S3 bucket for async output (defaults to input bucket)
            output_prefix: S3 prefix for async output files
            account_id: AWS account ID (auto-detected if not provided)
            s3_client: Optional pre-configured S3 client
            bedrock_client: Optional pre-configured Bedrock Runtime client
        """
        self.region = region
        self.output_bucket = output_bucket
        self.output_prefix = output_prefix

        config = Config(
            region_name=region,
            retries={"max_attempts": 3, "mode": "adaptive"}
        )

        self.s3_client = s3_client or boto3.client("s3", config=config)
        self.bedrock_client = bedrock_client or boto3.client(
            "bedrock-runtime",
            config=config
        )

        # Get account ID (required for S3 bucket owner)
        if account_id:
            self.account_id = account_id
        else:
            sts_client = boto3.client("sts", config=config)
            self.account_id = sts_client.get_caller_identity()["Account"]

    def get_video_embeddings(
        self,
        bucket: str,
        s3_key: str,
        embedding_types: Optional[list] = None,
        segment_length_sec: int = 6,
        account_id: Optional[str] = None
    ) -> dict:
        """
        Generate multi-vector embeddings for a video from S3 using async API.

        Args:
            bucket: S3 bucket name
            s3_key: S3 object key for the video file
            embedding_types: List of embedding types to generate.
                           Options: ["visual", "audio"]
                           Defaults to both.
            segment_length_sec: Length of each segment in seconds (default: 6)
            account_id: AWS account ID for bucket owner (optional)

        Returns:
            Dictionary containing:
            - segments: List of segment data with embeddings
            - metadata: Video metadata
        """
        if embedding_types is None:
            embedding_types = ["visual", "audio", "transcription"]

        s3_uri = f"s3://{bucket}/{s3_key}"
        output_bucket = self.output_bucket or bucket
        # Output URI should be a prefix/directory - Bedrock writes output files there
        output_prefix = f"{self.output_prefix}{s3_key.replace('/', '_')}/"
        output_uri = f"s3://{output_bucket}/{output_prefix}"

        # Prepare the request payload for async invocation (Marengo 3.0 format)
        # Note: Marengo 3.0 uses nested "video" structure and requires bucketOwner
        bucket_owner = account_id or self.account_id
        model_input = {
            "inputType": "video",
            "video": {
                "mediaSource": {
                    "s3Location": {
                        "uri": s3_uri,
                        "bucketOwner": bucket_owner
                    }
                },
                "segmentation": {
                    "method": "fixed",
                    "fixed": {
                        "durationSec": segment_length_sec
                    }
                },
                "embeddingOption": embedding_types,
                "embeddingScope": ["clip", "asset"]
            }
        }

        # Store output prefix for reading results later
        self._last_output_bucket = output_bucket
        self._last_output_prefix = output_prefix

        print(f"Starting async video embedding job...")
        print(f"Input: {s3_uri}")
        print(f"Output: {output_uri}")

        # Start async invocation
        response = self.bedrock_client.start_async_invoke(
            modelId=self.MODEL_ID,
            modelInput=model_input,
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": output_uri
                }
            }
        )

        invocation_arn = response["invocationArn"]
        print(f"Invocation ARN: {invocation_arn}")

        # Poll for completion
        result = self._wait_for_completion(invocation_arn)

        if result["status"] != "Completed":
            raise RuntimeError(
                f"Async invocation failed with status: {result['status']}. "
                f"Failure reason: {result.get('failureMessage', 'Unknown')}"
            )

        # Read results from S3 output directory
        embeddings_response = self._read_output_from_s3(output_bucket, output_prefix)

        return self._parse_embeddings_response(embeddings_response, s3_uri)

    def _wait_for_completion(self, invocation_arn: str) -> dict:
        """Poll for async invocation completion."""
        for attempt in range(self.MAX_POLL_ATTEMPTS):
            response = self.bedrock_client.get_async_invoke(
                invocationArn=invocation_arn
            )

            status = response["status"]
            print(f"Status: {status} (attempt {attempt + 1})")

            if status in ["Completed", "Failed"]:
                return response

            time.sleep(self.POLL_INTERVAL_SECONDS)

        raise TimeoutError(
            f"Async invocation did not complete within "
            f"{self.MAX_POLL_ATTEMPTS * self.POLL_INTERVAL_SECONDS} seconds"
        )

    def _read_output_from_s3(self, bucket: str, prefix: str) -> dict:
        """Read the async invocation output from S3 directory."""
        # List objects in the output prefix
        response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if "Contents" not in response:
            raise RuntimeError(f"No output files found in s3://{bucket}/{prefix}")

        # Find the output.json file (not manifest)
        output_key = None
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith("output.json") or (key.endswith(".json") and "manifest" not in key.lower()):
                output_key = key
                break

        if not output_key:
            # Try reading the first json file that's not a manifest
            for obj in response["Contents"]:
                key = obj["Key"]
                if key.endswith(".json") and "manifest" not in key.lower():
                    output_key = key
                    break

        if not output_key:
            available = [obj["Key"] for obj in response["Contents"]]
            raise RuntimeError(f"No output.json found. Available files: {available}")

        print(f"Reading output from: s3://{bucket}/{output_key}")
        response = self.s3_client.get_object(Bucket=bucket, Key=output_key)
        content = response["Body"].read().decode("utf-8")
        return json.loads(content)

    def get_text_query_embedding(self, query_text: str) -> dict:
        """
        Generate embedding for a text query (synchronous).

        Args:
            query_text: The search query text

        Returns:
            Dictionary containing the query embedding (512 dimensions)
        """
        # Marengo 3.0 text format - no embeddingOption for text queries
        request_body = {
            "inputType": "text",
            "text": {
                "inputText": query_text
            }
        }

        response = self.bedrock_client.invoke_model(
            modelId=self.MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )

        response_body = json.loads(response["body"].read())

        # Marengo 3.0 returns {"data": [{"embedding": [...]}]}
        embedding = []
        if "data" in response_body and response_body["data"]:
            embedding = response_body["data"][0].get("embedding", [])
        else:
            embedding = response_body.get("embedding", [])

        return {
            "embedding": embedding,
            "input_text": query_text
        }

    def decompose_query(self, query_text: str) -> dict:
        """
        Decompose a query into modality-specific sub-queries using Claude Haiku.

        Uses Claude Haiku 4.5 to intelligently decompose a user query into:
        - Visual query: Focuses on what appears on screen
        - Audio query: Focuses on sounds, music, audio elements
        - Transcription query: Focuses on spoken words, dialogue

        Args:
            query_text: The original search query

        Returns:
            Dictionary with decomposed queries:
            {
                "original_query": str,
                "visual": str,
                "audio": str,
                "transcription": str
            }
        """
        prompt = f"""You are a video search query decomposer for multi-modal video search. Your task is to extract and assign different parts of the query to the appropriate modality.

Given this search query: "{query_text}"

Decompose it into THREE modality-specific queries by extracting relevant parts:

1. VISUAL query: Extract visual elements (what appears on screen)
   - Include: people, objects, scenes, actions, settings, visual elements
   - Remove: audio descriptions and speech content

2. AUDIO query: Extract audio elements (NON-SPEECH sounds only)
   - Include: music, sound effects, ambient sounds, background audio
   - Remove: visual elements and speech content

3. TRANSCRIPTION query: Extract speech content (what is spoken/said)
   - Include: dialogue, narration, spoken words, verbal content
   - Remove: visual elements and non-speech audio

APPROACH:
- PRIMARY: Extract and assign parts of the original query to the appropriate modality
- SECONDARY: If a modality isn't explicitly mentioned, expand naturally based on context
- DO NOT just repeat the same query three times
- DO NOT invent content that contradicts the original query

Examples:

Query: "A basketball player dunking in an empty stadium with beats playing in the background while a narrator discusses the importance of high school basketball programs"
{{
    "visual": "basketball player dunking in an empty stadium",
    "audio": "beats playing in the background",
    "transcription": "narrator discusses the importance of high school basketball programs"
}}

Query: "explosion scene with loud bang"
{{
    "visual": "explosion, fire, debris, smoke",
    "audio": "loud bang, explosion sound",
    "transcription": "talking about explosion"
}}

Query: "Ross says I take thee Rachel at a wedding"
{{
    "visual": "Ross at a wedding ceremony, wedding scene",
    "audio": "wedding music, ceremony sounds",
    "transcription": "Ross says I take thee Rachel"
}}

Query: "person laughing at a joke"
{{
    "visual": "person laughing, smiling",
    "audio": "laughter sounds, giggling",
    "transcription": "telling a joke, funny conversation"
}}

Now decompose: "{query_text}"

Respond in JSON format ONLY:
{{
    "visual": "extracted visual elements",
    "audio": "extracted audio elements (non-speech)",
    "transcription": "extracted speech/dialogue content"
}}"""

        try:
            response = self.bedrock_client.invoke_model(
                modelId="us.anthropic.claude-haiku-4-5-20250929-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )

            response_body = json.loads(response["body"].read())

            # Extract the text content from Claude's response
            content = response_body.get("content", [])
            if content and len(content) > 0:
                text_response = content[0].get("text", "")

                # Parse the JSON from the response
                # Remove markdown code blocks if present
                text_response = text_response.strip()
                if text_response.startswith("```json"):
                    text_response = text_response[7:]
                if text_response.startswith("```"):
                    text_response = text_response[3:]
                if text_response.endswith("```"):
                    text_response = text_response[:-3]
                text_response = text_response.strip()

                decomposed = json.loads(text_response)

                return {
                    "original_query": query_text,
                    "visual": decomposed.get("visual", query_text),
                    "audio": decomposed.get("audio", query_text),
                    "transcription": decomposed.get("transcription", query_text)
                }
            else:
                # Fallback: use original query for all modalities
                return {
                    "original_query": query_text,
                    "visual": query_text,
                    "audio": query_text,
                    "transcription": query_text
                }

        except Exception as e:
            print(f"Query decomposition failed: {e}")
            # Fallback: use original query for all modalities
            return {
                "original_query": query_text,
                "visual": query_text,
                "audio": query_text,
                "transcription": query_text
            }

    def _parse_embeddings_response(self, response: dict, s3_uri: str) -> dict:
        """
        Parse the Bedrock Marengo response into structured segment data.

        Marengo 3.0 response format:
        {
            "data": [
                {
                    "embeddingOption": "visual",
                    "embeddingScope": "clip",
                    "startSec": 0.0,
                    "endSec": 6.0,
                    "embedding": [...]
                },
                ...
            ]
        }

        Args:
            response: Raw Bedrock response
            s3_uri: Original S3 URI for reference

        Returns:
            Structured embeddings data with segments grouped by time
        """
        # Marengo 3.0 uses "data" array with flattened structure
        raw_embeddings = response.get("data", response.get("embeddings", []))

        # Group embeddings by (startSec, endSec) to create segments
        # Only use "clip" scope for segment-level embeddings
        segment_map = {}

        for emb in raw_embeddings:
            scope = emb.get("embeddingScope", "clip")

            # Skip asset-level embeddings for segment grouping
            if scope == "asset":
                continue

            start_sec = emb.get("startSec", 0.0)
            end_sec = emb.get("endSec", 0.0)
            segment_key = (start_sec, end_sec)

            if segment_key not in segment_map:
                segment_map[segment_key] = {
                    "start_time": start_sec,
                    "end_time": end_sec,
                    "embeddings": {}
                }

            # Map embeddingOption to our modality names
            emb_option = emb.get("embeddingOption", "")
            embedding_vector = emb.get("embedding", [])

            if emb_option == "visual":
                segment_map[segment_key]["embeddings"]["visual"] = embedding_vector
            elif emb_option == "audio":
                segment_map[segment_key]["embeddings"]["audio"] = embedding_vector
            elif emb_option == "transcription":
                segment_map[segment_key]["embeddings"]["transcription"] = embedding_vector

        # Convert to sorted list of segments
        sorted_keys = sorted(segment_map.keys(), key=lambda x: x[0])
        segments = []

        for idx, key in enumerate(sorted_keys):
            segment_data = segment_map[key]
            segment_data["segment_id"] = idx
            segment_data["s3_uri"] = s3_uri
            segments.append(segment_data)

        return {
            "segments": segments,
            "metadata": {
                "s3_uri": s3_uri,
                "total_segments": len(segments),
                "embedding_dimension": self.EMBEDDING_DIMENSION,
                "model_id": self.MODEL_ID
            }
        }


def create_client(
    region: str = "us-east-1",
    output_bucket: Optional[str] = None
) -> BedrockMarengoClient:
    """Factory function to create a BedrockMarengoClient."""
    return BedrockMarengoClient(region=region, output_bucket=output_bucket)
