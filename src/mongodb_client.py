"""
MongoDB Atlas Client for Multi-Vector Video Embeddings

Handles storage and retrieval of visual, audio, and transcription
embeddings in modality-specific collections (multi-collection mode).

Each modality has its own collection and vector index:
- visual_embeddings / visual_embeddings_vector_index
- audio_embeddings / audio_embeddings_vector_index
- transcription_embeddings / transcription_embeddings_vector_index
"""

import os
from typing import Optional, List
from datetime import datetime
from pymongo import MongoClient
from pymongo.database import Database


FINGERPRINT_COLLECTION = "video_fingerprints"


class MongoDBEmbeddingClient:
    """Client for storing and querying multi-vector embeddings in MongoDB Atlas."""

    # Modality-specific collections
    MODALITY_COLLECTIONS = {
        "visual": "visual_embeddings",
        "audio": "audio_embeddings",
        "transcription": "transcription_embeddings"
    }

    # Vector index names (one per modality collection)
    MODALITY_INDEX_NAMES = {
        "visual": "visual_embeddings_vector_index",
        "audio": "audio_embeddings_vector_index",
        "transcription": "transcription_embeddings_vector_index"
    }

    # Valid modality types
    MODALITY_TYPES = ["visual", "audio", "transcription"]

    # Embedding dimension for Marengo 3.0
    EMBEDDING_DIMENSION = 512

    def __init__(
        self,
        connection_string: Optional[str] = None,
        database_name: str = "video_search"
    ):
        self.connection_string = connection_string or os.environ.get("MONGODB_URI")
        if not self.connection_string:
            raise ValueError(
                "MongoDB connection string required. "
                "Provide via parameter or MONGODB_URI environment variable."
            )

        self.database_name = database_name
        self.client = MongoClient(self.connection_string)
        self.db: Database = self.client[database_name]

    def store_segment_embeddings(
        self,
        video_id: str,
        segment_id: int,
        s3_uri: str,
        start_time: float,
        end_time: float,
        embeddings: dict,
    ) -> dict:
        """
        Store embeddings for a video segment across modality-specific collections.

        Args:
            video_id: Unique identifier for the video
            segment_id: Segment index within the video
            s3_uri: S3 URI of the source video
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            embeddings: Dict containing 'visual', 'audio', and/or 'transcription' embeddings

        Returns:
            Dictionary with inserted IDs for each modality
        """
        base_doc = {
            "video_id": video_id,
            "segment_id": segment_id,
            "s3_uri": s3_uri,
            "start_time": start_time,
            "end_time": end_time,
            "created_at": datetime.utcnow(),
        }

        inserted_ids = {}

        for modality in self.MODALITY_TYPES:
            if modality in embeddings and embeddings[modality]:
                doc = {
                    **base_doc,
                    "embedding": embeddings[modality]
                }
                collection_name = self.MODALITY_COLLECTIONS[modality]
                collection = self.db[collection_name]
                result = collection.insert_one(doc)
                inserted_ids[modality] = str(result.inserted_id)

        return inserted_ids

    def store_all_segments(self, video_id: str, segments: list) -> dict:
        """
        Store all segments from a video processing result.

        Args:
            video_id: Unique identifier for the video
            segments: List of segment dictionaries from BedrockMarengoClient

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

        for segment in segments:
            inserted = self.store_segment_embeddings(
                video_id=video_id,
                segment_id=segment["segment_id"],
                s3_uri=segment["s3_uri"],
                start_time=segment["start_time"],
                end_time=segment["end_time"],
                embeddings=segment.get("embeddings", {}),
            )

            results["segments_processed"] += 1
            if "visual" in inserted:
                results["visual_stored"] += 1
            if "audio" in inserted:
                results["audio_stored"] += 1
            if "transcription" in inserted:
                results["transcription_stored"] += 1

        return results

    def vector_search(
        self,
        query_embedding: list,
        modality: str,
        limit: int = 10,
        num_candidates: int = 100,
        video_id_filter: Optional[str] = None
    ) -> list:
        """
        Perform vector similarity search on a modality-specific collection.

        Args:
            query_embedding: Query embedding vector (512 dimensions)
            modality: Modality type ("visual", "audio", "transcription")
            limit: Maximum number of results to return
            num_candidates: Number of candidates for HNSW search
            video_id_filter: Filter by specific video ID

        Returns:
            List of matching documents with similarity scores
        """
        num_candidates = max(num_candidates, limit)

        if modality not in self.MODALITY_TYPES:
            raise ValueError(f"Invalid modality: {modality}")

        collection_name = self.MODALITY_COLLECTIONS[modality]
        index_name = self.MODALITY_INDEX_NAMES[modality]
        collection = self.db[collection_name]

        vector_search_filter = {}
        if video_id_filter:
            vector_search_filter["video_id"] = video_id_filter

        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": num_candidates,
                    "limit": limit,
                    **({"filter": vector_search_filter} if vector_search_filter else {})
                }
            },
            {
                "$project": {
                    "video_id": 1,
                    "segment_id": 1,
                    "s3_uri": 1,
                    "start_time": 1,
                    "end_time": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        results = list(collection.aggregate(pipeline))

        # Add modality_type for downstream consistency
        for result in results:
            result["modality_type"] = modality

        return results

    def multi_modality_search(
        self,
        query_embedding: list,
        limit_per_modality: int = 50,
        modalities: Optional[List[str]] = None,
        video_id_filter: Optional[str] = None,
    ) -> dict:
        """
        Search across multiple modalities and return results grouped by modality.

        Args:
            query_embedding: Query embedding vector
            limit_per_modality: Max results per modality
            modalities: List of modalities to search (default: all three)
            video_id_filter: Optional filter by video ID

        Returns:
            Dictionary with results grouped by modality type
        """
        if modalities is None:
            modalities = self.MODALITY_TYPES

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

    def delete_video_embeddings(self, video_id: str) -> dict:
        """
        Delete all embeddings for a specific video from all modality collections.

        Args:
            video_id: Video identifier

        Returns:
            Dictionary with deletion counts per collection
        """
        results = {}
        query = {"video_id": video_id}

        for modality, coll_name in self.MODALITY_COLLECTIONS.items():
            coll = self.db[coll_name]
            result = coll.delete_many(query)
            results[modality] = result.deleted_count

        results["total"] = sum(results.values())
        return results

    def get_collection_stats(self) -> dict:
        """Get document counts by modality collection."""
        counts = {}
        total = 0
        for modality, coll_name in self.MODALITY_COLLECTIONS.items():
            count = self.db[coll_name].count_documents({})
            counts[modality] = count
            total += count

        return {
            "total_documents": total,
            "by_modality": counts
        }

    def store_video_fingerprint(self, video_id: str, visual_fp: list, audio_fp: list,
                                transcription_fp: list, segment_count: int,
                                total_duration: float, video_name: str = "",
                                thumbnail_key: str = None,
                                technical_metadata: dict = None) -> bool:
        """Store or update a video fingerprint document."""
        collection = self.db[FINGERPRINT_COLLECTION]
        doc = {
            "video_id": video_id,
            "video_name": video_name,
            "visual_fingerprint": visual_fp,
            "audio_fingerprint": audio_fp,
            "transcription_fingerprint": transcription_fp,
            "segment_count": segment_count,
            "total_duration": total_duration,
            "thumbnail_key": thumbnail_key,
            "technical_metadata": technical_metadata,
            "created_at": datetime.utcnow()
        }
        result = collection.replace_one({"video_id": video_id}, doc, upsert=True)
        return result.acknowledged

    def get_video_fingerprint(self, video_id: str) -> Optional[dict]:
        """Get fingerprint for a single video."""
        collection = self.db[FINGERPRINT_COLLECTION]
        return collection.find_one({"video_id": video_id}, {"_id": 0})

    def get_all_fingerprints(self) -> list:
        """Get all video fingerprints for similarity comparison."""
        collection = self.db[FINGERPRINT_COLLECTION]
        return list(collection.find({}, {"_id": 0}))

    def delete_video_fingerprint(self, video_id: str) -> int:
        """Delete fingerprint for a video."""
        collection = self.db[FINGERPRINT_COLLECTION]
        result = collection.delete_one({"video_id": video_id})
        return result.deleted_count

    def get_segments_for_video(self, video_id: str) -> list:
        """Get all segment embeddings for a video from modality collections.
        Results are sorted by segment_id + modality for deterministic ordering."""
        segments = []
        for coll_name, modality in [
            ("visual_embeddings", "visual"),
            ("audio_embeddings", "audio"),
            ("transcription_embeddings", "transcription")
        ]:
            for doc in self.db[coll_name].find(
                {"video_id": video_id},
                {"_id": 0, "segment_id": 1, "embedding": 1, "start_time": 1, "end_time": 1, "s3_uri": 1}
            ).sort([("segment_id", 1)]):
                doc["modality_type"] = modality
                segments.append(doc)
        return segments

    def close(self):
        """Close the MongoDB connection."""
        self.client.close()


def create_client(
    connection_string: Optional[str] = None,
    database_name: str = "video_search"
) -> MongoDBEmbeddingClient:
    """Factory function to create a MongoDBEmbeddingClient."""
    return MongoDBEmbeddingClient(
        connection_string=connection_string,
        database_name=database_name
    )
