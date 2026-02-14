"""
S3 Vectors Client for Multi-Modal Video Search

Provides vector storage and search using Amazon S3 Vectors
with separate indexes for visual, audio, and transcription modalities.

This serves as the multi-index backend (alternative to MongoDB single-index).
"""

import math
import os
import boto3
from typing import Optional, List
from datetime import datetime


# AWS Profile for SSO authentication (local dev only, App Runner uses IAM roles)
AWS_PROFILE = os.environ.get("AWS_PROFILE")


class S3VectorsClient:
    """Client for storing and querying multi-modal embeddings in S3 Vectors."""

    # Unified index for all modalities (single-index mode)
    UNIFIED_INDEX_NAME = "unified-embeddings"

    # Index names for each modality (multi-index mode)
    INDEX_NAMES = {
        "visual": "visual-embeddings",
        "audio": "audio-embeddings",
        "transcription": "transcription-embeddings"
    }

    # Valid modality types
    MODALITY_TYPES = ["visual", "audio", "transcription"]

    # Embedding dimension for Marengo 3.0
    EMBEDDING_DIMENSION = 512

    # RRF constant (standard value used by Elasticsearch, etc.)
    RRF_K = 60

    # Default weights for fusion
    DEFAULT_WEIGHTS = {
        "visual": 0.8,
        "audio": 0.1,
        "transcription": 0.05
    }

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: str = "us-east-1",
        profile_name: Optional[str] = None
    ):
        """
        Initialize the S3 Vectors client.

        Args:
            bucket_name: Name of the S3 vector bucket (reads from S3_VECTORS_BUCKET env var if not provided).
            region: AWS region.
            profile_name: AWS profile to use (default from AWS_PROFILE env var).
        """
        # Read from environment variable if not provided
        if bucket_name is None:
            bucket_name = os.environ.get("S3_VECTORS_BUCKET", "your-vectors-bucket-name")

        self.bucket_name = bucket_name
        self.region = region

        # Use profile for SSO authentication (local dev) or default credentials (App Runner)
        profile = profile_name or AWS_PROFILE
        if profile:
            try:
                session = boto3.Session(profile_name=profile)
                self.client = session.client("s3vectors", region_name=region)
            except Exception:
                # Fall back to default credentials if profile fails
                self.client = boto3.client("s3vectors", region_name=region)
        else:
            # No profile specified - use default credentials (IAM role on App Runner)
            self.client = boto3.client("s3vectors", region_name=region)

    def store_segment_embeddings(
        self,
        video_id: str,
        segment_id: int,
        s3_uri: str,
        start_time: float,
        end_time: float,
        embeddings: dict,
        dual_write: bool = True
    ) -> dict:
        """
        Store embeddings for a video segment in S3 Vectors.

        Supports dual-write mode: writes to both unified-embeddings index
        and modality-specific indexes for single/multi-index search support.

        Args:
            video_id: Unique identifier for the video
            segment_id: Segment index within the video
            s3_uri: S3 URI of the source video
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            embeddings: Dict containing 'visual', 'audio', and/or 'transcription' embeddings
            dual_write: If True, write to both unified and modality-specific indexes

        Returns:
            Dictionary with status for each modality
        """
        results = {}

        # Write to modality-specific indexes (multi-index mode)
        for modality in self.MODALITY_TYPES:
            if modality not in embeddings or not embeddings[modality]:
                continue

            index_name = self.INDEX_NAMES[modality]
            vector_key = f"{video_id}_{segment_id}"

            # Prepare vector with metadata
            vector_data = {
                "key": vector_key,
                "data": {"float32": embeddings[modality]},
                "metadata": {
                    "video_id": video_id,
                    "segment_id": str(segment_id),
                    "s3_uri": s3_uri,
                    "start_time": str(start_time),
                    "end_time": str(end_time)
                }
            }

            try:
                self.client.put_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=index_name,
                    vectors=[vector_data]
                )
                results[modality] = "success"
            except Exception as e:
                results[modality] = f"error: {str(e)}"

        # Also write to unified index (single-index mode) if dual_write enabled
        if dual_write:
            for modality in self.MODALITY_TYPES:
                if modality not in embeddings or not embeddings[modality]:
                    continue

                # Unique key for unified index includes modality
                unified_key = f"{video_id}_{segment_id}_{modality}"

                vector_data = {
                    "key": unified_key,
                    "data": {"float32": embeddings[modality]},
                    "metadata": {
                        "video_id": video_id,
                        "segment_id": str(segment_id),
                        "modality_type": modality,  # Important: filter by this
                        "s3_uri": s3_uri,
                        "start_time": str(start_time),
                        "end_time": str(end_time)
                    }
                }

                try:
                    self.client.put_vectors(
                        vectorBucketName=self.bucket_name,
                        indexName=self.UNIFIED_INDEX_NAME,
                        vectors=[vector_data]
                    )
                except Exception as e:
                    # Don't overwrite success status if multi-index write succeeded
                    if results.get(modality) == "success":
                        results[f"{modality}_unified"] = f"error: {str(e)}"

        return results

    def store_all_segments(self, video_id: str, segments: list, dual_write: bool = True) -> dict:
        """
        Store all segments from a video processing result using batched API calls.

        This method batches put_vectors calls (up to 100 vectors per call) to dramatically
        reduce API latency overhead. For example, 213 segments × 3 modalities × 2 (dual write)
        = 1,278 vectors can be stored in ~13 batched calls instead of 1,278 individual calls.

        Args:
            video_id: Unique identifier for the video
            segments: List of segment dictionaries from BedrockMarengoClient
            dual_write: If True, write to both unified and modality-specific indexes

        Returns:
            Summary of stored segments
        """
        results = {
            "video_id": video_id,
            "segments_processed": 0,
            "visual_stored": 0,
            "audio_stored": 0,
            "transcription_stored": 0
        }

        # Collect all vectors to write, grouped by index
        modality_vectors = {modality: [] for modality in self.MODALITY_TYPES}
        unified_vectors = []

        # Build up vector lists for batching
        for segment in segments:
            segment_id = segment["segment_id"]
            s3_uri = segment["s3_uri"]
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            embeddings = segment.get("embeddings", {})

            results["segments_processed"] += 1

            # Add to modality-specific index vectors
            for modality in self.MODALITY_TYPES:
                if modality not in embeddings or not embeddings[modality]:
                    continue

                vector_key = f"{video_id}_{segment_id}"
                modality_vectors[modality].append({
                    "key": vector_key,
                    "data": {"float32": embeddings[modality]},
                    "metadata": {
                        "video_id": video_id,
                        "segment_id": str(segment_id),
                        "s3_uri": s3_uri,
                        "start_time": str(start_time),
                        "end_time": str(end_time)
                    }
                })

                # Also add to unified index if dual_write enabled
                if dual_write:
                    unified_key = f"{video_id}_{segment_id}_{modality}"
                    unified_vectors.append({
                        "key": unified_key,
                        "data": {"float32": embeddings[modality]},
                        "metadata": {
                            "video_id": video_id,
                            "segment_id": str(segment_id),
                            "modality_type": modality,
                            "s3_uri": s3_uri,
                            "start_time": str(start_time),
                            "end_time": str(end_time)
                        }
                    })

        # Write modality-specific indexes in batches of 100
        for modality in self.MODALITY_TYPES:
            vectors = modality_vectors[modality]
            if not vectors:
                continue

            index_name = self.INDEX_NAMES[modality]
            batch_size = 100

            # Split into batches and write
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                try:
                    self.client.put_vectors(
                        vectorBucketName=self.bucket_name,
                        indexName=index_name,
                        vectors=batch
                    )
                    results[f"{modality}_stored"] += len(batch)
                except Exception as e:
                    print(f"Error writing {modality} batch {i//batch_size + 1}: {e}")

        # Write unified index in batches of 100
        if dual_write and unified_vectors:
            batch_size = 100
            for i in range(0, len(unified_vectors), batch_size):
                batch = unified_vectors[i:i + batch_size]
                try:
                    self.client.put_vectors(
                        vectorBucketName=self.bucket_name,
                        indexName=self.UNIFIED_INDEX_NAME,
                        vectors=batch
                    )
                except Exception as e:
                    print(f"Error writing unified batch {i//batch_size + 1}: {e}")

        return results

    def vector_search(
        self,
        query_embedding: list,
        modality: str,
        limit: int = 50,
        video_id_filter: Optional[str] = None
    ) -> list:
        """
        Perform vector similarity search on a specific modality index.

        Args:
            query_embedding: Query embedding vector (512 dimensions)
            modality: Which modality index to search ("visual", "audio", "transcription")
            limit: Maximum number of results to return
            video_id_filter: Optional filter by video ID (uses native S3 Vectors metadata filtering)

        Returns:
            List of matching vectors with scores and metadata
        """
        if modality not in self.INDEX_NAMES:
            raise ValueError(f"Invalid modality: {modality}")

        index_name = self.INDEX_NAMES[modality]

        try:
            # Build query parameters (S3 Vectors API max topK is 100)
            query_params = {
                "vectorBucketName": self.bucket_name,
                "indexName": index_name,
                "queryVector": {"float32": query_embedding},
                "topK": min(limit, 100),
                "returnMetadata": True,
                "returnDistance": True
            }

            # Add native metadata filter if video_id is specified
            if video_id_filter:
                query_params["filter"] = {"video_id": {"$eq": video_id_filter}}

            response = self.client.query_vectors(**query_params)

            results = []
            for vector in response.get("vectors", []):
                metadata = vector.get("metadata", {})

                results.append({
                    "video_id": metadata.get("video_id", ""),
                    "segment_id": int(metadata.get("segment_id", 0)),
                    "s3_uri": metadata.get("s3_uri", ""),
                    "start_time": float(metadata.get("start_time", 0)),
                    "end_time": float(metadata.get("end_time", 0)),
                    "modality_type": modality,
                    # S3 Vectors returns SQUARED Euclidean distance for normalized vectors
                    # For normalized vectors: squared_euclidean = 2 * (1 - cosine_similarity)
                    # Therefore: cosine_similarity = 1 - (squared_euclidean / 2)
                    "score": 1 - (vector.get("distance", 0) / 2)
                })

            return results

        except Exception as e:
            print(f"S3 Vectors search error for {modality}: {e}")
            return []

    def multi_modality_search(
        self,
        query_embedding: list,
        limit_per_modality: int = 50,
        modalities: Optional[List[str]] = None,
        video_id_filter: Optional[str] = None,
        use_multi_index: bool = True
    ) -> dict:
        """
        Search across multiple modalities and return results grouped by modality.

        Args:
            query_embedding: Query embedding vector
            limit_per_modality: Max results per modality
            modalities: List of modalities to search (default: all three)
            video_id_filter: Optional filter by video ID
            use_multi_index: If True, use modality-specific indexes;
                           if False, use unified index with metadata filtering

        Returns:
            Dictionary with results grouped by modality type
        """
        if modalities is None:
            modalities = self.MODALITY_TYPES

        if use_multi_index:
            # Use modality-specific indexes (multi-index mode)
            results = {}
            for modality in modalities:
                if modality in self.MODALITY_TYPES:
                    results[modality] = self.vector_search(
                        query_embedding=query_embedding,
                        modality=modality,
                        limit=limit_per_modality,
                        video_id_filter=video_id_filter
                    )
            return results
        else:
            # Use unified index with modality filtering (single-index mode)
            return self.unified_multi_modality_search(
                query_embedding=query_embedding,
                limit_per_modality=limit_per_modality,
                modalities=modalities,
                video_id_filter=video_id_filter
            )

    def unified_multi_modality_search(
        self,
        query_embedding: list,
        limit_per_modality: int = 50,
        modalities: Optional[List[str]] = None,
        video_id_filter: Optional[str] = None
    ) -> dict:
        """
        Search across modalities using the unified index with native metadata filtering.

        Uses S3 Vectors native metadata filtering to search each modality separately
        within the unified index. This avoids the topK=100 limitation of post-filtering
        and ensures consistent results across all modalities.

        Args:
            query_embedding: Query embedding vector
            limit_per_modality: Max results per modality
            modalities: List of modalities to search
            video_id_filter: Optional filter by video ID

        Returns:
            Dictionary with results grouped by modality type
        """
        if modalities is None:
            modalities = self.MODALITY_TYPES

        results = {modality: [] for modality in modalities}

        # Search each modality separately using native metadata filtering
        for modality in modalities:
            try:
                # Build metadata filter
                # Filter by modality_type (required) and optionally by video_id
                metadata_filter = {"modality_type": {"$eq": modality}}

                if video_id_filter:
                    # Combine with video_id filter using $and
                    metadata_filter = {
                        "$and": [
                            {"modality_type": {"$eq": modality}},
                            {"video_id": {"$eq": video_id_filter}}
                        ]
                    }

                # Query unified index with metadata filter (S3 Vectors API max topK is 100)
                response = self.client.query_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=self.UNIFIED_INDEX_NAME,
                    queryVector={"float32": query_embedding},
                    topK=min(limit_per_modality, 100),
                    filter=metadata_filter,  # Native S3 Vectors metadata filtering
                    returnMetadata=True,
                    returnDistance=True
                )

                # Process results for this modality
                for vector in response.get("vectors", []):
                    metadata = vector.get("metadata", {})

                    results[modality].append({
                        "video_id": metadata.get("video_id", ""),
                        "segment_id": int(metadata.get("segment_id", 0)),
                        "s3_uri": metadata.get("s3_uri", ""),
                        "start_time": float(metadata.get("start_time", 0)),
                        "end_time": float(metadata.get("end_time", 0)),
                        "modality_type": modality,
                        "score": 1 - (vector.get("distance", 0) / 2)
                    })

            except Exception as e:
                print(f"S3 Vectors unified search error for {modality}: {e}")

        return results

    def get_index_stats(self) -> dict:
        """Get vector counts for each modality index."""
        stats = {}
        for modality, index_name in self.INDEX_NAMES.items():
            try:
                response = self.client.get_index(
                    vectorBucketName=self.bucket_name,
                    indexName=index_name
                )
                # Note: S3 Vectors doesn't provide vector count directly
                # We'd need to list vectors to count them
                stats[modality] = {
                    "index_name": index_name,
                    "status": response.get("status", "unknown")
                }
            except Exception as e:
                stats[modality] = {"error": str(e)}

        return stats

    def delete_video_embeddings(self, video_id: str) -> dict:
        """
        Delete all embeddings for a specific video from all indexes
        (unified + 3 modality-specific).

        Args:
            video_id: Video identifier

        Returns:
            Dictionary with deletion status per index
        """
        results = {}

        # Include unified index + modality-specific indexes
        all_indexes = {"unified": self.UNIFIED_INDEX_NAME}
        all_indexes.update(self.INDEX_NAMES)

        for label, index_name in all_indexes.items():
            try:
                deleted_keys = []
                next_token = None

                # Paginate through all vectors to find matches
                while True:
                    kwargs = dict(
                        vectorBucketName=self.bucket_name,
                        indexName=index_name,
                        maxResults=1000,
                    )
                    if next_token:
                        kwargs["nextToken"] = next_token

                    response = self.client.list_vectors(**kwargs)

                    for vector in response.get("vectors", []):
                        key = vector.get("key", "")
                        if key.startswith(f"{video_id}_"):
                            deleted_keys.append(key)

                    next_token = response.get("nextToken")
                    if not next_token:
                        break

                # Delete in batches of 100
                for i in range(0, len(deleted_keys), 100):
                    batch = deleted_keys[i:i + 100]
                    self.client.delete_vectors(
                        vectorBucketName=self.bucket_name,
                        indexName=index_name,
                        keys=batch,
                    )

                results[label] = {"deleted_count": len(deleted_keys)}

            except Exception as e:
                results[label] = {"error": str(e)}

        return results


    def search_with_fusion(
        self,
        query_embedding: list,
        modalities: Optional[List[str]] = None,
        weights: Optional[dict] = None,
        limit: int = 50,
        video_id_filter: Optional[str] = None,
        fusion_method: str = "rrf",
        use_multi_index: bool = True
    ) -> list:
        """
        Search across modalities with fusion (RRF or weighted).

        Args:
            query_embedding: Query embedding vector (512 dimensions)
            modalities: List of modalities to search
            weights: Weights per modality
            limit: Maximum results
            video_id_filter: Optional filter by video ID
            fusion_method: "rrf" or "weighted"
            use_multi_index: True = modality-specific indexes, False = unified index

        Returns:
            List of fused results with fusion scores
        """
        if modalities is None:
            modalities = self.MODALITY_TYPES

        if weights is None:
            weights = self.DEFAULT_WEIGHTS.copy()

        # Search each modality
        modality_results = self.multi_modality_search(
            query_embedding=query_embedding,
            limit_per_modality=limit * 2,
            modalities=modalities,
            video_id_filter=video_id_filter,
            use_multi_index=use_multi_index  # Pass index mode
        )

        # Apply fusion
        if fusion_method == "rrf":
            return self._rrf_fusion(modality_results, weights, limit)
        else:
            return self._weighted_fusion(modality_results, weights, limit)

    def _rrf_fusion(
        self,
        modality_results: dict,
        weights: dict,
        limit: int
    ) -> list:
        """
        Reciprocal Rank Fusion (RRF).

        Formula: score(d) = Σ w_m / (k + rank_m(d))
        """
        segment_scores = {}

        for modality, results in modality_results.items():
            weight = weights.get(modality, 1.0)

            for rank, doc in enumerate(results, start=1):
                key = (doc["video_id"], doc["start_time"])

                if key not in segment_scores:
                    segment_scores[key] = {
                        "video_id": doc["video_id"],
                        "segment_id": doc.get("segment_id", 0),
                        "start_time": doc["start_time"],
                        "end_time": doc["end_time"],
                        "s3_uri": doc["s3_uri"],
                        "rrf_score": 0.0,
                        "modality_scores": {},
                        "modality_ranks": {}
                    }

                # RRF contribution: weight / (k + rank)
                rrf_contribution = weight / (self.RRF_K + rank)
                segment_scores[key]["rrf_score"] += rrf_contribution
                segment_scores[key]["modality_scores"][modality] = doc["score"]
                segment_scores[key]["modality_ranks"][modality] = rank

        # Sort by RRF score
        ranked = sorted(
            segment_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # Normalize RRF scores to 0-1 range for better interpretability
        # Maximum possible RRF score is when all ranks = 1
        total_weight = sum(weights.values())
        max_rrf_score = total_weight / (self.RRF_K + 1)

        # Normalize and rename for API consistency
        for item in ranked:
            raw_rrf = item.pop("rrf_score")
            # Normalize to 0-1 range
            item["fusion_score"] = min(1.0, raw_rrf / max_rrf_score)

            # Confidence score = best modality match (for more representative UI display)
            if item.get("modality_scores"):
                item["confidence_score"] = max(item["modality_scores"].values())
            else:
                item["confidence_score"] = item["fusion_score"]

        return ranked[:limit]

    def _weighted_fusion(
        self,
        modality_results: dict,
        weights: dict,
        limit: int
    ) -> list:
        """
        Simple weighted score fusion.

        Formula: score(d) = Σ w_m * sim_m(d)
        """
        segment_scores = {}

        for modality, results in modality_results.items():
            for doc in results:
                key = (doc["video_id"], doc["start_time"])

                if key not in segment_scores:
                    segment_scores[key] = {
                        "video_id": doc["video_id"],
                        "segment_id": doc.get("segment_id", 0),
                        "start_time": doc["start_time"],
                        "end_time": doc["end_time"],
                        "s3_uri": doc["s3_uri"],
                        "modality_scores": {}
                    }

                segment_scores[key]["modality_scores"][modality] = doc["score"]

        # Compute weighted sum
        # CRITICAL FIX: Only sum weights for modalities that were actually searched
        # This prevents single-modality searches from being diluted by other modality weights
        searched_modalities = list(modality_results.keys())
        total_weight = sum(weights.get(m, 0) for m in searched_modalities)

        # Avoid division by zero
        if total_weight == 0:
            total_weight = len(searched_modalities)

        for key, data in segment_scores.items():
            fusion_score = sum(
                (weights.get(m, 0) / total_weight) * data["modality_scores"].get(m, 0)
                for m in searched_modalities
            )
            data["fusion_score"] = fusion_score

            # Confidence score = best modality match (for more representative UI display)
            if data["modality_scores"]:
                data["confidence_score"] = max(data["modality_scores"].values())
            else:
                data["confidence_score"] = fusion_score

        ranked = sorted(
            segment_scores.values(),
            key=lambda x: x["fusion_score"],
            reverse=True
        )

        return ranked[:limit]


def create_client(
    bucket_name: Optional[str] = None,
    region: str = "us-east-1"
) -> S3VectorsClient:
    """Factory function to create an S3VectorsClient. Reads from S3_VECTORS_BUCKET env var if bucket_name not provided."""
    return S3VectorsClient(bucket_name=bucket_name, region=region)
