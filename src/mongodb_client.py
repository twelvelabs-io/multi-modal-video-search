"""
MongoDB Atlas Client for Multi-Vector Video Embeddings

Handles storage and retrieval of visual, audio, and transcription
embeddings in a single collection with modality_type field for filtering.

Single-collection approach allows:
- Pre-filtering by modality_type to search specific modalities
- Searching all modalities in one query
- Flexible fusion strategies (weighted or anchor-based)
"""

import os
from typing import Optional, List
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


FINGERPRINT_COLLECTION = "video_fingerprints"


class MongoDBEmbeddingClient:
    """Client for storing and querying multi-vector embeddings in MongoDB Atlas."""

    # Single collection for all modalities (unified-embeddings)
    COLLECTION_NAME = "unified-embeddings"

    # Modality-specific collections for multi-index mode
    MODALITY_COLLECTIONS = {
        "visual": "visual_embeddings",
        "audio": "audio_embeddings",
        "transcription": "transcription_embeddings"
    }

    # Vector index names
    VECTOR_INDEX_NAME = "unified_embeddings_vector_index"  # For unified collection
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
        """
        Initialize the MongoDB client.

        Args:
            connection_string: MongoDB Atlas connection string.
                             If not provided, reads from MONGODB_URI env var.
            database_name: Name of the database to use.
        """
        self.connection_string = connection_string or os.environ.get("MONGODB_URI")
        if not self.connection_string:
            raise ValueError(
                "MongoDB connection string required. "
                "Provide via parameter or MONGODB_URI environment variable."
            )

        self.database_name = database_name
        self.client = MongoClient(self.connection_string)
        self.db: Database = self.client[database_name]
        self.collection: Collection = self.db[self.COLLECTION_NAME]

    def store_segment_embeddings(
        self,
        video_id: str,
        segment_id: int,
        s3_uri: str,
        start_time: float,
        end_time: float,
        embeddings: dict,
        dual_write: bool = False,
        index_modes: list = None
    ) -> dict:
        """
        Store embeddings for a video segment.

        Args:
            video_id: Unique identifier for the video
            segment_id: Segment index within the video
            s3_uri: S3 URI of the source video
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            embeddings: Dict containing 'visual', 'audio', and/or 'transcription' embeddings
            dual_write: Legacy param — ignored if index_modes is provided
            index_modes: List of modes to write to: ["single"], ["multi"], or ["single", "multi"]

        Returns:
            Dictionary with inserted IDs for each modality
        """
        # Resolve write targets from index_modes (preferred) or dual_write (legacy)
        if index_modes is not None:
            write_single = "single" in index_modes
            write_multi = "multi" in index_modes
        else:
            write_single = True  # dual_write=False means single only
            write_multi = dual_write

        base_doc = {
            "video_id": video_id,
            "segment_id": segment_id,
            "s3_uri": s3_uri,
            "start_time": start_time,
            "end_time": end_time,
            "created_at": datetime.utcnow(),
        }

        inserted_ids = {}
        documents_to_insert = []

        # Create a document for each modality that has embeddings
        for modality in self.MODALITY_TYPES:
            if modality in embeddings and embeddings[modality]:
                doc = {
                    **base_doc,
                    "modality_type": modality,
                    "embedding": embeddings[modality]
                }
                documents_to_insert.append((modality, doc))

        # Write to unified-embeddings collection (single index)
        if write_single and documents_to_insert:
            docs = [doc for _, doc in documents_to_insert]
            result = self.collection.insert_many(docs)

            for i, (modality, _) in enumerate(documents_to_insert):
                inserted_ids[modality] = str(result.inserted_ids[i])

        # Write to modality-specific collections (multi index)
        if write_multi and documents_to_insert:
            for modality, doc in documents_to_insert:
                collection_name = self.MODALITY_COLLECTIONS[modality]
                collection = self.db[collection_name]

                modality_doc = {k: v for k, v in doc.items() if k != 'modality_type'}

                result = collection.insert_one(modality_doc)
                inserted_ids[f"{modality}_multi"] = str(result.inserted_id)

        return inserted_ids

    def store_all_segments(self, video_id: str, segments: list, dual_write: bool = True, index_modes: list = None) -> dict:
        """
        Store all segments from a video processing result.

        Args:
            video_id: Unique identifier for the video
            segments: List of segment dictionaries from BedrockMarengoClient
            dual_write: Legacy — ignored if index_modes provided
            index_modes: List of modes: ["single"], ["multi"], or ["single", "multi"]

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
                dual_write=dual_write,
                index_modes=index_modes
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
        limit: int = 10,
        num_candidates: int = 100,
        modality_filter: Optional[str] = None,
        video_id_filter: Optional[str] = None
    ) -> list:
        # MongoDB requires numCandidates >= limit
        num_candidates = max(num_candidates, limit)
        """
        Perform vector similarity search with optional modality filtering.

        Args:
            query_embedding: Query embedding vector (512 dimensions)
            limit: Maximum number of results to return
            num_candidates: Number of candidates for HNSW search
            modality_filter: Filter by modality type ("visual", "audio", "transcription")
            video_id_filter: Filter by specific video ID

        Returns:
            List of matching documents with similarity scores
        """
        # Build filter for vector search
        vector_search_filter = {}
        if modality_filter:
            vector_search_filter["modality_type"] = modality_filter
        if video_id_filter:
            vector_search_filter["video_id"] = video_id_filter

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.VECTOR_INDEX_NAME,
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
                    "modality_type": 1,
                    "s3_uri": 1,
                    "start_time": 1,
                    "end_time": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        return list(self.collection.aggregate(pipeline))

    def multi_modality_search(
        self,
        query_embedding: list,
        limit_per_modality: int = 50,
        modalities: Optional[List[str]] = None,
        video_id_filter: Optional[str] = None,
        use_multi_index: bool = False
    ) -> dict:
        """
        Search across multiple modalities and return results grouped by modality.

        Args:
            query_embedding: Query embedding vector
            limit_per_modality: Max results per modality
            modalities: List of modalities to search (default: all three)
            video_id_filter: Optional filter by video ID
            use_multi_index: If True, search modality-specific collections;
                           if False, search unified collection with filters

        Returns:
            Dictionary with results grouped by modality type
        """
        if modalities is None:
            modalities = self.MODALITY_TYPES

        results = {}

        if use_multi_index:
            # Search modality-specific collections
            for modality in modalities:
                if modality in self.MODALITY_TYPES:
                    results[modality] = self.vector_search_multi_index(
                        query_embedding=query_embedding,
                        modality=modality,
                        limit=limit_per_modality,
                        video_id_filter=video_id_filter
                    )
        else:
            # Search unified collection with modality filters
            for modality in modalities:
                if modality in self.MODALITY_TYPES:
                    results[modality] = self.vector_search(
                        query_embedding=query_embedding,
                        limit=limit_per_modality,
                        modality_filter=modality,
                        video_id_filter=video_id_filter
                    )

        return results

    def vector_search_multi_index(
        self,
        query_embedding: list,
        modality: str,
        limit: int = 10,
        num_candidates: int = 100,
        video_id_filter: Optional[str] = None
    ) -> list:
        # MongoDB requires numCandidates >= limit
        num_candidates = max(num_candidates, limit)
        """
        Perform vector search on a modality-specific collection.

        Args:
            query_embedding: Query embedding vector (512 dimensions)
            modality: Modality type ("visual", "audio", "transcription")
            limit: Maximum number of results to return
            num_candidates: Number of candidates for HNSW search
            video_id_filter: Filter by specific video ID

        Returns:
            List of matching documents with similarity scores
        """
        if modality not in self.MODALITY_TYPES:
            raise ValueError(f"Invalid modality: {modality}")

        collection_name = self.MODALITY_COLLECTIONS[modality]
        index_name = self.MODALITY_INDEX_NAMES[modality]
        collection = self.db[collection_name]

        # Build filter
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

        # Add modality_type to match unified collection format
        for result in results:
            result["modality_type"] = modality

        return results

    def search_all_modalities(
        self,
        query_embedding: list,
        limit: int = 50,
        video_id_filter: Optional[str] = None
    ) -> list:
        """
        Search all modalities without filtering (for fusion in application layer).

        Args:
            query_embedding: Query embedding vector
            limit: Maximum total results
            video_id_filter: Optional filter by video ID

        Returns:
            List of all matching documents across modalities
        """
        return self.vector_search(
            query_embedding=query_embedding,
            limit=limit,
            num_candidates=limit * 3,  # More candidates since we're searching all
            modality_filter=None,
            video_id_filter=video_id_filter
        )

    def delete_video_embeddings(self, video_id: str) -> dict:
        """
        Delete all embeddings for a specific video from unified + modality collections.

        Args:
            video_id: Video identifier

        Returns:
            Dictionary with deletion counts per collection
        """
        results = {}
        query = {"video_id": video_id}

        # Unified collection
        result = self.collection.delete_many(query)
        results["unified"] = result.deleted_count

        # Modality-specific collections
        for modality, coll_name in self.MODALITY_COLLECTIONS.items():
            coll = self.db[coll_name]
            result = coll.delete_many(query)
            results[modality] = result.deleted_count

        results["total"] = sum(results.values())
        return results

    def get_collection_stats(self) -> dict:
        """Get document counts by modality type."""
        pipeline = [
            {"$group": {"_id": "$modality_type", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]

        counts = {modality: 0 for modality in self.MODALITY_TYPES}
        for doc in self.collection.aggregate(pipeline):
            if doc["_id"] in counts:
                counts[doc["_id"]] = doc["count"]

        return {
            "total_documents": self.collection.count_documents({}),
            "by_modality": counts
        }

    def store_video_fingerprint(self, video_id: str, visual_fp: list, audio_fp: list,
                                transcription_fp: list, segment_count: int,
                                total_duration: float, video_name: str = "",
                                thumbnail_key: str = None) -> bool:
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
        """Get all segment embeddings for a video from unified collection."""
        collection = self.db[self.COLLECTION_NAME]
        return list(collection.find(
            {"video_id": video_id},
            {"_id": 0, "segment_id": 1, "modality_type": 1, "embedding": 1,
             "start_time": 1, "end_time": 1, "s3_uri": 1}
        ))

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
