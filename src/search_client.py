"""
Multi-Modal Video Search Client

Performs fusion search across visual, audio, and transcription
embeddings stored in MongoDB Atlas or S3 Vectors.

Supports multiple fusion methods:
1. Reciprocal Rank Fusion (RRF) - rank-based fusion, more robust
2. Weighted Score Fusion - simple weighted sum
3. Dynamic Intent Routing - auto-calculates weights based on query intent

Backend options:
- MongoDB Atlas: Single collection with modality_type filter (default)
- S3 Vectors: Separate indexes per modality (multi-index mode)

RRF formula: score(d) = Σ w_m / (k + rank_m(d))
"""

import math
import os
from typing import Optional
from pymongo import MongoClient

from bedrock_client import BedrockMarengoClient
from s3_vectors_client import S3VectorsClient
from mongodb_client import MongoDBEmbeddingClient


# Anchor text prompts for dynamic intent routing (Section 4.3 of whitepaper)
ANCHOR_PROMPTS = {
    "visual": "What appears on screen: people, objects, scenes, actions, clothing, colors, and visual composition of the video.",
    "audio": "The non-speech audio in the video: music, sound effects, ambient sound, and other audio elements.",
    "transcription": "The spoken words in the video: dialogue, narration, speech, and what people say."
}


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def softmax_with_temperature(scores: dict, temperature: float = 10.0) -> dict:
    """Apply softmax with temperature to get weights."""
    # Handle empty input - return empty dict
    if not scores:
        return {}

    # Scale by temperature
    scaled = {k: v * temperature for k, v in scores.items()}
    # Compute max for numerical stability
    max_val = max(scaled.values())
    # Compute exp
    exp_scores = {k: math.exp(v - max_val) for k, v in scaled.items()}
    # Normalize
    total = sum(exp_scores.values())
    return {k: v / total for k, v in exp_scores.items()}


class VideoSearchClient:
    """Client for multi-modal video search with fusion."""

    # TwelveLabs-style weights: heavily favor visual for most queries
    DEFAULT_WEIGHTS = {
        "visual": 0.8,
        "audio": 0.1,
        "transcription": 0.05
    }

    # RRF constant (standard value used by Elasticsearch, etc.)
    RRF_K = 60

    # Temperature for softmax in dynamic routing
    SOFTMAX_TEMPERATURE = 10.0

    # Modality-specific collection names for multi-index mode
    MODALITY_COLLECTIONS = {
        "visual": "visual_embeddings",
        "audio": "audio_embeddings",
        "transcription": "transcription_embeddings"
    }

    def __init__(
        self,
        mongodb_uri: str,
        database_name: str = "video_search",
        collection_name: str = "video_embeddings",
        index_name: str = "vector_index",
        bedrock_region: str = "us-east-1"
    ):
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client[database_name]
        self.collection = self.db[collection_name]
        self.index_name = index_name
        self.bedrock = BedrockMarengoClient(
            region=bedrock_region,
            output_bucket=os.environ.get("S3_BUCKET", "your-media-bucket-name")
        )
        # Anchor embeddings for dynamic routing (lazy initialized)
        self._anchor_embeddings = None

        # Multi-index collections (lazy initialized)
        self._modality_collections = None

        # S3 Vectors client for multi-index mode (lazy initialized)
        self._s3_vectors_client = None

        # MongoDB embedding client (lazy initialized)
        self._mongodb_client = None
        self._mongodb_uri = mongodb_uri
        self._database_name = database_name

    def get_s3_vectors_client(self) -> S3VectorsClient:
        """Get or create the S3 Vectors client."""
        if self._s3_vectors_client is None:
            self._s3_vectors_client = S3VectorsClient()
        return self._s3_vectors_client

    def get_mongodb_client(self) -> MongoDBEmbeddingClient:
        """Get or create the MongoDB embedding client."""
        if self._mongodb_client is None:
            self._mongodb_client = MongoDBEmbeddingClient(
                connection_string=self._mongodb_uri,
                database_name=self._database_name
            )
        return self._mongodb_client

    def has_s3_vectors_backend(self) -> bool:
        """Check if S3 Vectors backend is available."""
        try:
            client = self.get_s3_vectors_client()
            stats = client.get_index_stats()
            return all("error" not in v for v in stats.values())
        except Exception:
            return False

    def get_modality_collection(self, modality: str):
        """Get the collection for a specific modality (multi-index mode)."""
        collection_name = self.MODALITY_COLLECTIONS.get(modality)
        if not collection_name:
            raise ValueError(f"Unknown modality: {modality}")
        return self.db[collection_name]

    def has_multi_index_collections(self) -> bool:
        """Check if multi-index collections exist and have data."""
        for modality in self.MODALITY_COLLECTIONS:
            coll = self.get_modality_collection(modality)
            if coll.count_documents({}, limit=1) == 0:
                return False
        return True

    def initialize_anchors(self) -> dict:
        """
        Pre-compute anchor embeddings for dynamic intent routing.
        Call this at app startup to avoid latency on first query.
        """
        if self._anchor_embeddings is not None:
            return self._anchor_embeddings

        self._anchor_embeddings = {}
        for modality, prompt in ANCHOR_PROMPTS.items():
            result = self.bedrock.get_text_query_embedding(prompt)
            self._anchor_embeddings[modality] = result["embedding"]

        return self._anchor_embeddings

    def get_anchor_embeddings(self) -> dict:
        """Get anchor embeddings, initializing if needed."""
        if self._anchor_embeddings is None:
            self.initialize_anchors()
        return self._anchor_embeddings

    def compute_dynamic_weights(
        self,
        query_embedding: list,
        temperature: float = None
    ) -> dict:
        """
        Compute dynamic weights based on query intent (Section 4.3).

        Uses cosine similarity between query and anchor embeddings,
        then applies softmax with temperature to get weights.
        """
        if temperature is None:
            temperature = self.SOFTMAX_TEMPERATURE

        anchors = self.get_anchor_embeddings()

        # Compute cosine similarities
        similarities = {}
        for modality, anchor_emb in anchors.items():
            similarities[modality] = cosine_similarity(query_embedding, anchor_emb)

        # Apply softmax with temperature
        weights = softmax_with_temperature(similarities, temperature)

        return {
            "weights": weights,
            "similarities": similarities
        }

    def search(
        self,
        query: str,
        query_image: Optional[str] = None,  # Base64-encoded image
        modalities: Optional[list] = None,
        weights: Optional[dict] = None,
        limit: int = 50,
        video_id: Optional[str] = None,
        fusion_method: str = "rrf",  # "rrf", "weighted", or "dynamic"
        k_per_modality: int = 20,
        backend: str = "s3vectors",  # "mongodb" or "s3vectors"
        use_multi_index: bool = True,  # True = multi-index, False = single unified index
        return_embeddings: bool = False,  # Include 512d embeddings in results
        decomposed_queries: Optional[dict] = None  # LLM-decomposed queries per modality
    ) -> list:
        """
        Search for video segments matching a text query, image query, or both.

        Supports three query modes:
        1. Text only: query provided, query_image is None
        2. Image only: query_image provided, query is empty/None
        3. Image + Text: Both provided (multimodal search)

        Args:
            query: Text search query (optional if image provided)
            query_image: Base64-encoded image for image-to-video search (optional)
            modalities: List of modalities to search ["visual", "audio", "transcription"]
            weights: Weights per modality (for RRF, these weight the rank contribution)
            limit: Maximum results
            video_id: Optional filter by specific video
            fusion_method: "rrf" (Reciprocal Rank Fusion) or "weighted" (score sum)
            backend: "mongodb" or "s3vectors"
            use_multi_index: True = modality-specific indexes, False = unified index
            return_embeddings: If True, include 512d embedding vectors in results
            decomposed_queries: Optional dict with modality-specific queries (text-only)

        Returns:
            List of ranked results with fusion scores
        """
        if modalities is None:
            modalities = ["visual", "audio", "transcription"]

        if weights is None:
            weights = self.DEFAULT_WEIGHTS.copy()

        # Generate query embeddings (per modality if decomposed, otherwise shared)
        query_embeddings = {}

        if decomposed_queries:
            # Use decomposed queries to generate separate embeddings per modality
            print("Using LLM-decomposed queries:")
            for modality in modalities:
                decomposed_query = decomposed_queries.get(modality, query)
                print(f"  {modality}: {decomposed_query}")

                # If image is provided, combine decomposed text with image
                if query_image:
                    result = self.bedrock.get_multimodal_query_embedding(
                        query_text=decomposed_query,
                        query_image_base64=query_image
                    )
                else:
                    result = self.bedrock.get_text_query_embedding(decomposed_query)

                query_embeddings[modality] = result["embedding"]
        elif query_image:
            # Image-to-video or Image+Text-to-video search
            query_result = self.bedrock.get_multimodal_query_embedding(
                query_text=query if query else None,
                query_image_base64=query_image
            )
            shared_embedding = query_result["embedding"]

            if not shared_embedding:
                return []

            # Use same image embedding for all modalities
            for modality in modalities:
                query_embeddings[modality] = shared_embedding
        else:
            # Text-only search - use same query embedding for all modalities
            query_result = self.bedrock.get_text_query_embedding(query)
            shared_embedding = query_result["embedding"]

            if not shared_embedding:
                return []

            for modality in modalities:
                query_embeddings[modality] = shared_embedding

        # Route to correct backend
        if backend == "s3vectors":
            s3v_client = self.get_s3_vectors_client()

            # Search each modality with its specific embedding
            modality_results = {}
            for modality in modalities:
                weight = weights.get(modality, 1.0)
                if weight == 0:
                    continue

                query_embedding = query_embeddings.get(modality)
                if not query_embedding:
                    continue

                # Search this modality with its specific embedding
                results = s3v_client.multi_modality_search(
                    query_embedding=query_embedding,
                    limit_per_modality=limit * 2,
                    modalities=[modality],
                    video_id_filter=video_id,
                    use_multi_index=use_multi_index
                )

                modality_results[modality] = results.get(modality, [])

            # Apply fusion
            if fusion_method == "rrf":
                return self._rrf_fusion(modality_results, weights, limit)
            else:
                return self._weighted_fusion(modality_results, weights, limit)

        # MongoDB backend
        elif backend == "mongodb":
            # Use MongoDB client's multi_modality_search
            # NOTE: MongoDB currently only supports single-index mode (unified-embeddings)
            # due to free tier limit (3 search indexes max). Force use_multi_index=False.
            mongo_client = self.get_mongodb_client()

            # For each modality, search with its specific embedding
            modality_results = {}
            for modality in modalities:
                weight = weights.get(modality, 1.0)
                if weight == 0:
                    continue

                query_embedding = query_embeddings.get(modality)
                if not query_embedding:
                    continue

                # Search this modality (always use single-index mode for MongoDB)
                results = mongo_client.multi_modality_search(
                    query_embedding=query_embedding,
                    limit_per_modality=limit * 2,
                    modalities=[modality],
                    video_id_filter=video_id,
                    use_multi_index=False  # Force single-index mode for MongoDB
                )

                modality_results[modality] = results.get(modality, [])

            # Apply fusion
            if fusion_method == "rrf":
                return self._rrf_fusion(modality_results, weights, limit)
            else:
                return self._weighted_fusion(modality_results, weights, limit)

        else:
            raise ValueError(f"Invalid backend: {backend}")

    def _rrf_fusion(
        self,
        modality_results: dict,
        weights: dict,
        limit: int
    ) -> list:
        """
        Reciprocal Rank Fusion (RRF).

        Formula: score(d) = Σ w_m / (k + rank_m(d))

        This is more robust than score-based fusion because:
        - Handles different score distributions across modalities
        - Emphasizes agreement between modalities
        - Standard approach used by Elasticsearch, etc.
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
                    # Preserve embedding if present
                    if "embedding" in doc:
                        segment_scores[key]["embedding"] = doc["embedding"]

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
                    # Preserve embedding if present
                    if "embedding" in doc:
                        segment_scores[key]["embedding"] = doc["embedding"]

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

    def search_dynamic(
        self,
        query: str,
        query_image: Optional[str] = None,
        limit: int = 50,
        video_id: Optional[str] = None,
        temperature: float = None,
        backend: str = "s3vectors",
        use_multi_index: bool = True,
        return_embeddings: bool = False,
        decomposed_queries: Optional[dict] = None
    ) -> dict:
        """
        Search with dynamic intent-based routing (Section 4.3 of whitepaper).

        Automatically determines modality weights based on query semantics.
        Supports text, image, or image+text queries.

        Args:
            query: Text search query (optional if image provided)
            query_image: Base64-encoded image for image-to-video search (optional)
            limit: Maximum results
            video_id: Optional filter by specific video
            temperature: Softmax temperature (higher = more uniform weights)
            backend: "mongodb" or "s3vectors"
            use_multi_index: True = modality-specific indexes, False = unified index
            return_embeddings: If True, include 512d embedding vectors in results
            decomposed_queries: Optional dict with modality-specific queries (text-only)

        Returns:
            Dict with 'results', 'weights', 'similarities', and optionally 'query_embedding'
        """
        if temperature is None:
            temperature = self.SOFTMAX_TEMPERATURE

        # Generate query embedding for dynamic weight computation
        if query_image:
            query_result = self.bedrock.get_multimodal_query_embedding(
                query_text=query if query else None,
                query_image_base64=query_image
            )
        else:
            query_result = self.bedrock.get_text_query_embedding(query)

        query_embedding = query_result["embedding"]

        if not query_embedding:
            return {"results": [], "weights": {}, "similarities": {}}

        # Compute dynamic weights based on query intent
        dynamic_result = self.compute_dynamic_weights(query_embedding, temperature)
        weights = dynamic_result["weights"]
        similarities = dynamic_result["similarities"]

        modalities = ["visual", "audio", "transcription"]

        # Generate modality-specific embeddings if decomposed queries provided
        query_embeddings = {}
        if decomposed_queries:
            print("Dynamic mode using LLM-decomposed queries:")
            for modality in modalities:
                decomposed_query = decomposed_queries.get(modality, query)
                print(f"  {modality}: {decomposed_query}")

                # If image is provided, combine decomposed text with image
                if query_image:
                    result = self.bedrock.get_multimodal_query_embedding(
                        query_text=decomposed_query,
                        query_image_base64=query_image
                    )
                else:
                    result = self.bedrock.get_text_query_embedding(decomposed_query)

                query_embeddings[modality] = result["embedding"]
        else:
            # Use same embedding for all modalities
            for modality in modalities:
                query_embeddings[modality] = query_embedding

        # Route to correct backend
        if backend == "s3vectors":
            # Search each modality with its specific embedding
            modality_results = {}
            s3v_client = self.get_s3_vectors_client()
            for modality in modalities:
                results = s3v_client.multi_modality_search(
                    query_embedding=query_embeddings[modality],
                    limit_per_modality=limit * 2,
                    modalities=[modality],
                    video_id_filter=video_id,
                    use_multi_index=use_multi_index
                )
                modality_results[modality] = results.get(modality, [])

            # Apply weighted fusion with dynamic weights
            results = self._weighted_fusion(modality_results, weights, limit)

            response = {
                "results": results,
                "weights": weights,
                "similarities": similarities
            }
            if return_embeddings:
                response["query_embedding"] = query_embedding
            return response

        # MongoDB backend
        elif backend == "mongodb":
            # NOTE: MongoDB uses single-index mode only (unified-embeddings)
            modality_results = {}
            mongo_client = self.get_mongodb_client()

            for modality in modalities:
                # Search this modality with its specific embedding
                results = mongo_client.multi_modality_search(
                    query_embedding=query_embeddings[modality],
                    limit_per_modality=limit * 2,
                    modalities=[modality],
                    video_id_filter=video_id,
                    use_multi_index=False  # Force single-index mode for MongoDB
                )
                modality_results[modality] = results.get(modality, [])

        # Apply weighted fusion with dynamic weights
        results = self._weighted_fusion(modality_results, weights, limit)

        response = {
            "results": results,
            "weights": weights,
            "similarities": similarities
        }
        if return_embeddings:
            response["query_embedding"] = query_embedding

        return response

    def get_videos(self) -> list:
        """Get list of all indexed videos."""
        pipeline = [
            {"$group": {"_id": "$video_id", "s3_uri": {"$first": "$s3_uri"}}},
            {"$project": {"video_id": "$_id", "s3_uri": 1, "_id": 0}}
        ]
        return list(self.collection.aggregate(pipeline))

    def get_segment(self, video_id: str, segment_id: int) -> Optional[dict]:
        """Get a specific segment by video_id and segment_id."""
        return self.collection.find_one(
            {"video_id": video_id, "segment_id": segment_id},
            {"video_id": 1, "segment_id": 1, "s3_uri": 1, "start_time": 1, "end_time": 1}
        )

    def close(self):
        """Close MongoDB connection."""
        self.mongo_client.close()


def create_client(
    mongodb_uri: str,
    database_name: str = "video_search"
) -> VideoSearchClient:
    """Factory function to create a VideoSearchClient."""
    return VideoSearchClient(mongodb_uri=mongodb_uri, database_name=database_name)
