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
from functools import lru_cache
from typing import Optional, Callable, Any
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 32.0,
    exponential_base: float = 2.0
) -> Any:
    """
    Retry a function with exponential backoff for transient errors.

    Handles Bedrock transient errors: ModelErrorException and InternalServerException.
    These occur due to temporary service issues and should be retried.
    """
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return func()
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')

            # Retry on transient Bedrock errors
            if error_code in ['ModelErrorException', 'InternalServerException']:
                if attempt == max_retries - 1:
                    # Last attempt, re-raise the error
                    raise

                # Log and wait before retrying
                print(f"Bedrock {error_code} (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s...")
                time.sleep(delay)

                # Exponential backoff with cap
                delay = min(delay * exponential_base, max_delay)
            else:
                # Non-retryable error, re-raise immediately
                raise

    # Should never reach here, but just in case
    return func()


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
        service_role_arn: Optional[str] = None,
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
            service_role_arn: IAM role ARN for Bedrock to assume for S3 access
            s3_client: Optional pre-configured S3 client
            bedrock_client: Optional pre-configured Bedrock Runtime client
        """
        self.region = region
        self.output_bucket = output_bucket
        self.output_prefix = output_prefix
        self.service_role_arn = service_role_arn

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
            # Try to get from environment variable first
            self.account_id = os.environ.get("AWS_ACCOUNT_ID")
            if not self.account_id:
                # Fallback to STS if not in environment
                sts_client = boto3.client("sts", config=config)
                self.account_id = sts_client.get_caller_identity()["Account"]

        # Embedding cache to avoid redundant API calls
        # Cache key: query text -> embedding vector
        # Limited to 1000 entries to prevent unbounded memory growth
        self._embedding_cache = {}
        self._cache_max_size = 1000

    def get_video_embeddings(
        self,
        bucket: str,
        s3_key: str,
        embedding_types: Optional[list] = None,
        segmentation_method: str = "dynamic",
        segment_length_sec: int = 6,
        min_duration_sec: int = 4,
        account_id: Optional[str] = None
    ) -> dict:
        """
        Generate multi-vector embeddings for a video from S3 using async API.

        Args:
            bucket: S3 bucket name
            s3_key: S3 object key for the video file
            embedding_types: List of embedding types to generate.
                           Options: ["visual", "audio", "transcription"]
                           Defaults to all three.
            segmentation_method: Segmentation method ("fixed" or "dynamic")
                               - "fixed": Fixed-length segments
                               - "dynamic": Shot boundary detection (default)
            segment_length_sec: Length of each segment in seconds for fixed method (default: 6)
            min_duration_sec: Minimum segment duration for dynamic method (1-5 seconds, default: 4)
            account_id: AWS account ID for bucket owner (optional)

        Returns:
            Dictionary containing:
            - segments: List of segment data with embeddings
            - metadata: Video metadata
        """
        if embedding_types is None:
            embedding_types = ["visual", "audio", "transcription"]

        # Do NOT URL-encode - Bedrock expects raw S3 URIs with spaces
        s3_uri = f"s3://{bucket}/{s3_key}"

        output_bucket = self.output_bucket or bucket
        # Output URI should be a prefix/directory - Bedrock writes output files there
        output_prefix = f"{self.output_prefix}{s3_key.replace('/', '_').replace(' ', '_')}/"
        output_uri = f"s3://{output_bucket}/{output_prefix}"

        # Prepare the request payload for async invocation (Marengo 3.0 format)
        # Note: Marengo 3.0 uses nested "video" structure and requires bucketOwner
        bucket_owner = account_id or self.account_id

        # Configure segmentation based on method
        if segmentation_method == "dynamic":
            # Dynamic shot boundary detection (1-5 seconds)
            segmentation_config = {
                "method": "dynamic",
                "dynamic": {
                    "minDurationSec": max(1, min(5, min_duration_sec))  # Clamp to valid range
                }
            }
        else:
            # Fixed-length segments
            segmentation_config = {
                "method": "fixed",
                "fixed": {
                    "durationSec": segment_length_sec
                }
            }

        model_input = {
            "inputType": "video",
            "video": {
                "mediaSource": {
                    "s3Location": {
                        "uri": s3_uri,
                        "bucketOwner": bucket_owner
                    }
                },
                "segmentation": segmentation_config,
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
        print(f"BucketOwner: {bucket_owner}")
        print(f"ModelInput: {json.dumps(model_input, indent=2)}")

        # Start async invocation
        # Note: Bedrock accesses S3 via bucket policy, not serviceRoleArn parameter
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

    def get_multimodal_query_embedding(
        self,
        query_text: str = None,
        query_image_base64: str = None
    ) -> dict:
        """
        Generate embedding for a multimodal query (text, image, or both).

        Supports three query modes:
        1. Text only: query_text provided, query_image_base64 is None
        2. Image only: query_image_base64 provided, query_text is None
        3. Image + Text: Both provided (multimodal search)

        Embeddings are cached by a hash of the inputs to avoid redundant API calls.

        Args:
            query_text: Optional text query
            query_image_base64: Optional base64-encoded image (JPEG, PNG, GIF, WebP, max 5MB)

        Returns:
            Dictionary containing the query embedding (512 dimensions)

        Raises:
            ValueError: If neither text nor image is provided
        """
        if not query_text and not query_image_base64:
            raise ValueError("At least one of query_text or query_image_base64 must be provided")

        # Create cache key from inputs
        cache_key = f"text:{query_text or ''}_image:{query_image_base64[:50] if query_image_base64 else ''}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Build request body based on input types
        if query_image_base64 and query_text:
            # Multimodal: Image + Text (uses text_image input type)
            request_body = {
                "inputType": "text_image",
                "text_image": {
                    "inputText": query_text,
                    "mediaSource": {
                        "base64String": query_image_base64
                    }
                }
            }
        elif query_image_base64:
            # Image only (uses image input type)
            request_body = {
                "inputType": "image",
                "image": {
                    "mediaSource": {
                        "base64String": query_image_base64
                    }
                }
            }
        else:
            # Text only (fallback to original behavior)
            request_body = {
                "inputType": "text",
                "text": {
                    "inputText": query_text
                }
            }

        # Wrap invoke_model with retry logic to handle transient errors
        def _invoke():
            return self.bedrock_client.invoke_model(
                modelId=self.MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )

        response = retry_with_exponential_backoff(_invoke)
        response_body = json.loads(response["body"].read())

        # Marengo 3.0 returns {"data": [{"embedding": [...]}]}
        embedding = []
        if "data" in response_body and response_body["data"]:
            embedding = response_body["data"][0].get("embedding", [])
        else:
            embedding = response_body.get("embedding", [])

        result = {
            "embedding": embedding,
            "text": query_text,
            "has_image": bool(query_image_base64)
        }

        # Cache the result
        self._embedding_cache[cache_key] = result
        if len(self._embedding_cache) > self._cache_max_size:
            # Evict oldest 100 entries
            keys_to_remove = list(self._embedding_cache.keys())[:100]
            for key in keys_to_remove:
                del self._embedding_cache[key]

        return result

    def get_text_query_embedding(self, query_text: str) -> dict:
        """
        Generate embedding for a text query (synchronous).

        Embeddings are cached by query text to avoid redundant API calls.
        When switching between fusion methods, backends, or index modes with
        the same query, the cached embedding is reused.

        Args:
            query_text: The search query text

        Returns:
            Dictionary containing the query embedding (512 dimensions)
        """
        # Check cache first
        if query_text in self._embedding_cache:
            return self._embedding_cache[query_text]

        # Marengo 3.0 text format - no embeddingOption for text queries
        request_body = {
            "inputType": "text",
            "text": {
                "inputText": query_text
            }
        }

        # Wrap invoke_model with retry logic to handle transient errors
        def _invoke():
            return self.bedrock_client.invoke_model(
                modelId=self.MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )

        response = retry_with_exponential_backoff(_invoke)
        response_body = json.loads(response["body"].read())

        # Marengo 3.0 returns {"data": [{"embedding": [...]}]}
        embedding = []
        if "data" in response_body and response_body["data"]:
            embedding = response_body["data"][0].get("embedding", [])
        else:
            embedding = response_body.get("embedding", [])

        result = {
            "embedding": embedding,
            "input_text": query_text
        }

        # Cache the result (with size limit to prevent unbounded growth)
        if len(self._embedding_cache) >= self._cache_max_size:
            # Simple FIFO eviction: remove oldest 10% of entries
            keys_to_remove = list(self._embedding_cache.keys())[:100]
            for key in keys_to_remove:
                del self._embedding_cache[key]

        self._embedding_cache[query_text] = result

        return result

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
        prompt = f"""You are a video search query decomposer. Your task is to split a user's search query into THREE different modality-specific queries.

Original query: "{query_text}"

CRITICAL: Each modality query MUST be different from the others. DO NOT repeat the same text.

Generate THREE DISTINCT queries:

1. VISUAL query - What appears ON SCREEN:
   - Focus on: people, objects, scenes, actions, clothing, colors, settings
   - Extract visual elements from the original query
   - Expand with relevant visual context if needed

2. AUDIO query - NON-SPEECH sounds ONLY:
   - Focus on: music, sound effects, ambient noise, background audio
   - Extract audio elements from the original query
   - If no audio mentioned, infer relevant sounds for the scene

3. TRANSCRIPTION query - SPOKEN WORDS:
   - Focus on: dialogue, narration, speech, what people say
   - Extract speech/text elements from the original query
   - If no speech mentioned, infer what might be discussed

EXAMPLES:

Input: "A basketball player dunking in an empty stadium with beats playing in the background while a narrator discusses the importance of high school basketball programs"
Output:
{{
    "visual": "basketball player dunking in an empty stadium",
    "audio": "beats playing in the background",
    "transcription": "narrator discusses the importance of high school basketball programs"
}}

Input: "Ross says I take thee Rachel at a wedding"
Output:
{{
    "visual": "Ross at a wedding ceremony, wedding altar, formal attire",
    "audio": "wedding music, ceremony sounds, emotional atmosphere",
    "transcription": "Ross says I take thee Rachel"
}}

Input: "explosion scene"
Output:
{{
    "visual": "explosion, fire, debris, smoke, destruction",
    "audio": "loud bang, explosion sound, rumbling",
    "transcription": "describing explosion, emergency response"
}}

Input: "person laughing"
Output:
{{
    "visual": "person laughing, smiling face, happy expression",
    "audio": "laughter sounds, giggling, chuckling",
    "transcription": "telling jokes, humorous conversation"
}}

Now decompose this query: "{query_text}"

Return ONLY valid JSON in this exact format:
{{
    "visual": "your visual query here",
    "audio": "your audio query here",
    "transcription": "your transcription query here"
}}"""

        try:
            # Wrap invoke_model with retry logic to handle transient errors
            def _invoke():
                return self.bedrock_client.invoke_model(
                    modelId="anthropic.claude-3-haiku-20240307-v1:0",
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 500,
                        "temperature": 0.3,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                )

            response = retry_with_exponential_backoff(_invoke)
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

    # Pegasus video analysis
    PEGASUS_MODEL_ID = "us.twelvelabs.pegasus-1-2-v1:0"

    def analyze_video(self, s3_uri: str, prompt: str, start_time: float = None, end_time: float = None) -> str:
        """
        Call TwelveLabs Pegasus to analyze a video with a natural language prompt.

        Args:
            s3_uri: S3 URI of the video (e.g., s3://bucket/proxies/video.mp4)
            prompt: Natural language question about the video
            start_time: Optional segment start time in seconds
            end_time: Optional segment end time in seconds

        Returns:
            Text response from Pegasus
        """
        full_prompt = prompt
        if start_time is not None and end_time is not None:
            full_prompt = f"Focus on the segment from {start_time:.1f}s to {end_time:.1f}s. {prompt}"

        body = {
            "inputPrompt": full_prompt,
            "mediaSource": {
                "s3Location": {
                    "uri": s3_uri,
                    "bucketOwner": self.account_id
                }
            },
            "temperature": 0.2,
            "maxOutputTokens": 2048
        }

        response = retry_with_exponential_backoff(
            lambda: self.bedrock_client.invoke_model(
                modelId=self.PEGASUS_MODEL_ID,
                body=json.dumps(body)
            )
        )
        result = json.loads(response["body"].read())
        return result.get("message", "")


def create_client(
    region: str = "us-east-1",
    output_bucket: Optional[str] = None
) -> BedrockMarengoClient:
    """Factory function to create a BedrockMarengoClient."""
    return BedrockMarengoClient(region=region, output_bucket=output_bucket)
