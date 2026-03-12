"""
Video Search API

FastAPI backend for multi-modal video search with Bedrock Marengo
and MongoDB Atlas vector search.
"""

import json
import logging
import os
import sys
import uuid
from typing import Optional
from urllib.parse import urlparse

# Configure logging so chat_agent INFO messages appear in App Runner logs
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

import boto3
import botocore.config
from fastapi import FastAPI, Query, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse, StreamingResponse
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from search_client import VideoSearchClient
from chat_agent import ChatAgent
from compare_client import compute_fingerprint, find_similar_videos, align_segments

# Configuration (set via environment variables)
MONGODB_URI = os.environ.get("MONGODB_URI")
S3_BUCKET = os.environ.get("S3_BUCKET", "your-media-bucket-name")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
CLOUDFRONT_DOMAIN = os.environ.get("CLOUDFRONT_DOMAIN", "xxxxx.cloudfront.net")

OLD_S3_PREFIX = "s3://tl-brice-media/WBD_project/Videos/proxy/"
NEW_S3_PREFIX = f"s3://{S3_BUCKET}/proxies/"


def _normalize_s3_uri(s3_uri: str) -> str:
    """Normalize old S3 URIs to current bucket/path."""
    if s3_uri.startswith(OLD_S3_PREFIX):
        filename = s3_uri[len(OLD_S3_PREFIX):]
        return NEW_S3_PREFIX + filename
    return s3_uri


# Initialize FastAPI
app = FastAPI(
    title="Video Search API",
    description="Multi-modal video search using Bedrock Marengo and MongoDB Atlas",
    version="1.0.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Search client (lazy init)
_search_client = None


def get_search_client() -> VideoSearchClient:
    """Get or create search client."""
    global _search_client
    if _search_client is None:
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI environment variable is required")
        _search_client = VideoSearchClient(
            mongodb_uri=MONGODB_URI,
            bedrock_region=AWS_REGION
        )
    return _search_client


# Chat agent (lazy init)
_chat_agent = None


def get_chat_agent() -> ChatAgent:
    """Get or create chat agent."""
    global _chat_agent
    if _chat_agent is None:
        search_client = get_search_client()
        _chat_agent = ChatAgent(
            bedrock_runtime_client=search_client.bedrock.bedrock_client,
            search_client=search_client,
            cloudfront_domain=CLOUDFRONT_DOMAIN
        )
    return _chat_agent


class SearchRequest(BaseModel):
    """Search request body."""
    query: Optional[str] = ""  # Optional for image-only searches
    query_image: Optional[str] = None  # Base64-encoded image for image-to-video search
    modalities: Optional[list] = None
    weights: Optional[dict] = None
    limit: int = 50
    video_id: Optional[str] = None
    fusion_method: str = "rrf"  # "rrf", "weighted", or "dynamic"
    temperature: Optional[float] = 10.0  # For dynamic mode
    backend: str = "s3vectors"  # "mongodb" or "s3vectors"
    use_multi_index: bool = True  # True = modality-specific indexes, False = unified index
    use_decomposition: bool = False  # True = use LLM to decompose query per modality


class SearchResult(BaseModel):
    """Single search result."""
    video_id: str
    segment_id: int
    start_time: float
    end_time: float
    s3_uri: str
    fusion_score: float
    modality_scores: dict
    modality_ranks: Optional[dict] = None  # For RRF mode
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None


class DynamicSearchResponse(BaseModel):
    """Response for dynamic search with computed weights."""
    results: list
    computed_weights: dict
    anchor_similarities: dict


@app.on_event("startup")
async def startup_event():
    """Initialize anchor embeddings at startup for dynamic routing."""
    try:
        client = get_search_client()
        client.initialize_anchors()
        print("Anchor embeddings initialized for dynamic routing")
    except Exception as e:
        print(f"Warning: Failed to initialize anchors at startup: {e}")


@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to prevent 404."""
    return Response(content=b"", media_type="image/x-icon")


@app.get("/api/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


@app.get("/api/index-mode")
async def get_index_mode():
    """Check available index modes (MongoDB single-index vs S3 Vectors multi-index)."""
    client = get_search_client()
    has_s3_vectors = client.has_s3_vectors_backend()
    return {
        "mongodb_available": True,
        "s3_vectors_available": has_s3_vectors,
        "default": "mongodb",
        # Legacy fields for backward compatibility
        "single_index_available": True,
        "multi_index_available": has_s3_vectors
    }


@app.post("/api/search")
async def search(request: SearchRequest):
    """
    Search for video segments.

    - **query**: Text search query
    - **modalities**: List of modalities ["visual", "audio", "transcription"]
    - **weights**: Custom weights per modality
    - **limit**: Max results (default 50)
    - **video_id**: Filter by specific video
    - **use_decomposition**: Use LLM to decompose query per modality
    """
    try:
        client = get_search_client()

        # LLM Query Decomposition (if enabled and text query provided)
        decomposed_queries = None
        if request.use_decomposition and request.query:
            decomposed_queries = client.bedrock.decompose_query(request.query)

        # Get query embedding (supports text, image, or both)
        if request.query_image:
            # Image-to-video or Image+Text-to-video search
            query_embedding_result = client.bedrock.get_multimodal_query_embedding(
                query_text=request.query if request.query else None,
                query_image_base64=request.query_image
            )
        else:
            # Text-only search
            query_embedding_result = client.bedrock.get_text_query_embedding(request.query)

        query_embedding = query_embedding_result.get("embedding", [])
    except Exception as e:
        print(f"Search error (initialization): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    try:
        results = client.search(
            query=request.query,
            query_image=request.query_image,  # Pass image for multimodal search
            modalities=request.modalities,
            weights=request.weights,
            limit=request.limit,
            video_id=request.video_id,
            fusion_method=request.fusion_method,
            backend=request.backend,  # Pass backend selection
            use_multi_index=request.use_multi_index,  # Pass index mode
            return_embeddings=True,  # Request embeddings in results
            decomposed_queries=decomposed_queries  # Pass decomposed queries if available
        )
    except Exception as e:
        print(f"Search error (backend): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    try:
        # Normalize S3 URIs and add CloudFront URLs
        for result in results:
            s3_uri = _normalize_s3_uri(result["s3_uri"])
            result["s3_uri"] = s3_uri
            parsed = urlparse(s3_uri)
            key = parsed.path.lstrip("/")

            # Map s3_uri key to proxies/ path for CloudFront delivery
            if "/proxies/" in key or key.startswith("proxies/"):
                proxy_key = key
            elif "WBD_project/Videos/proxy/" in key:
                # Backward compat: old data stored with legacy path
                proxy_key = "proxies/" + key.split("WBD_project/Videos/proxy/", 1)[1]
            elif key.startswith("input/"):
                proxy_key = key.replace("input/", "proxies/", 1)
            else:
                proxy_key = key
            result["video_url"] = f"https://{CLOUDFRONT_DOMAIN}/{proxy_key}"

            # Thumbnail URL (we'll generate these separately)
            result["thumbnail_url"] = f"/api/thumbnail/{result['video_id']}/{result['segment_id']}"

        response = {
            "results": results,
            "query_embeddings": {
                "combined": query_embedding  # 512d vector
            }
        }

        # Include decomposed queries if enabled
        if decomposed_queries:
            response["decomposed_queries"] = decomposed_queries

        return response
    except Exception as e:
        print(f"Search error (results processing): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get("/api/search")
async def search_get(
    q: str = Query(..., description="Search query"),
    modalities: Optional[str] = Query(None, description="Comma-separated modalities"),
    limit: int = Query(50, description="Max results"),
    video_id: Optional[str] = Query(None, description="Filter by video ID")
) -> list[SearchResult]:
    """GET version of search for simple queries."""
    mod_list = modalities.split(",") if modalities else None
    return await search(SearchRequest(
        query=q,
        modalities=mod_list,
        limit=limit,
        video_id=video_id
    ))


@app.post("/api/search/dynamic")
async def search_dynamic(request: SearchRequest):
    """
    Search with dynamic intent-based routing.

    Automatically determines modality weights based on query semantics.
    Returns computed weights alongside results.

    - **query**: Text search query
    - **limit**: Max results (default 50)
    - **video_id**: Filter by specific video
    - **temperature**: Softmax temperature (default 10.0, higher = more uniform)
    - **use_decomposition**: Use LLM to decompose query per modality
    """
    try:
        client = get_search_client()

        # LLM Query Decomposition (if enabled and text query provided)
        decomposed_queries = None
        if request.use_decomposition and request.query:
            decomposed_queries = client.bedrock.decompose_query(request.query)

        response = client.search_dynamic(
            query=request.query,
            query_image=request.query_image,  # Pass image for multimodal search
            limit=request.limit,
            video_id=request.video_id,
            temperature=request.temperature,
            backend=request.backend,  # Pass backend selection
            use_multi_index=request.use_multi_index,  # Pass index mode
            return_embeddings=True,
            decomposed_queries=decomposed_queries
        )

        # Debug logging for weights
        print(f"Dynamic search weights: {response.get('weights', {})}")
        print(f"Dynamic search similarities: {response.get('similarities', {})}")

        results = response["results"]
        query_embedding = response.get("query_embedding", [])

        # Normalize S3 URIs and add CloudFront URLs
        for result in results:
            s3_uri = _normalize_s3_uri(result["s3_uri"])
            result["s3_uri"] = s3_uri
            parsed = urlparse(s3_uri)
            key = parsed.path.lstrip("/")

            # Map s3_uri key to proxies/ path for CloudFront delivery
            if "/proxies/" in key or key.startswith("proxies/"):
                proxy_key = key
            elif "WBD_project/Videos/proxy/" in key:
                proxy_key = "proxies/" + key.split("WBD_project/Videos/proxy/", 1)[1]
            elif key.startswith("input/"):
                proxy_key = key.replace("input/", "proxies/", 1)
            else:
                proxy_key = key
            result["video_url"] = f"https://{CLOUDFRONT_DOMAIN}/{proxy_key}"
            result["thumbnail_url"] = f"/api/thumbnail/{result['video_id']}/{result['segment_id']}"

        api_response = {
            "results": results,
            "computed_weights": response["weights"],
            "anchor_similarities": response["similarities"],
            "query_embeddings": {
                "combined": query_embedding
            }
        }

        # Include decomposed queries if enabled
        if decomposed_queries:
            api_response["decomposed_queries"] = decomposed_queries

        return api_response
    except Exception as e:
        print(f"Dynamic search error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.get("/api/videos")
async def list_videos():
    """List all indexed videos."""
    client = get_search_client()
    videos = client.get_videos()

    # Add CloudFront URLs
    for video in videos:
        s3_uri = video.get("s3_uri", "")
        if s3_uri:
            parsed = urlparse(s3_uri)
            key = parsed.path.lstrip("/")
            video["video_url"] = f"https://{CLOUDFRONT_DOMAIN}/{key}"

    return videos


@app.get("/api/indexes/{backend}/{index_mode}/videos")
async def list_index_videos(backend: str, index_mode: str):
    """List videos in a specific index with cluster groupings."""
    client = get_search_client()

    if backend == "s3vectors":
        try:
            s3v_client = client.get_s3_vectors_client()
            if index_mode == "unified":
                index_name = s3v_client.UNIFIED_INDEX_NAME
            else:
                index_name = s3v_client.INDEX_NAMES.get("visual", "visual-embeddings")
            videos = s3v_client.list_videos(index_name=index_name)
        except Exception as e:
            print(f"Error listing S3 Vectors videos: {e}")
            return {"videos": [], "clusters": []}
    elif backend == "mongodb":
        db = client.db
        if index_mode == "unified":
            collection = db["unified-embeddings"]
        else:
            collection = db["visual_embeddings"]

        pipeline = [
            {"$group": {
                "_id": "$video_id",
                "s3_uri": {"$first": "$s3_uri"},
                "segment_count": {"$sum": 1}
            }},
            {"$project": {"video_id": "$_id", "s3_uri": 1, "segment_count": 1, "_id": 0}},
            {"$sort": {"video_id": 1}}
        ]
        videos = list(collection.aggregate(pipeline))
    else:
        return {"videos": [], "clusters": []}

    # Add CloudFront URLs and human-readable names
    for video in videos:
        s3_uri = video.get("s3_uri", "")
        if s3_uri:
            s3_uri = _normalize_s3_uri(s3_uri)
            video["s3_uri"] = s3_uri

            parsed = urlparse(s3_uri)
            key = parsed.path.lstrip("/")
            if key.startswith("input/"):
                key = key.replace("input/", "proxies/", 1)
            video["video_url"] = f"https://{CLOUDFRONT_DOMAIN}/{key}"

            # Thumbnail URL: proxy key with _thumb.jpg suffix
            thumb_key = os.path.splitext(key)[0] + "_thumb.jpg"
            video["thumbnail_url"] = f"https://{CLOUDFRONT_DOMAIN}/{thumb_key}"

            filename = os.path.basename(key)
            name_no_ext = os.path.splitext(filename)[0]
            video["name"] = name_no_ext.replace("_", " ").replace("-", " ")
        else:
            video["name"] = video.get("video_id", "Unknown")

    # --- Clustering ---
    # Fetch average visual embedding per video for clustering
    # NOTE: S3 Vectors backend does not support embedding retrieval for clustering yet.
    # Clusters will be empty for S3 Vectors — this is a known limitation.
    import numpy as np
    video_embeddings = {}
    try:
        if backend == "mongodb":
            # Use visual_embeddings collection (or unified with modality_type filter)
            if index_mode == "multi":
                emb_collection = db["visual_embeddings"]
                emb_pipeline = [
                    {"$group": {
                        "_id": "$video_id",
                        "avg_embedding": {"$push": "$embedding"}
                    }}
                ]
            else:
                emb_collection = db["unified-embeddings"]
                emb_pipeline = [
                    {"$match": {"modality_type": "visual"}},
                    {"$group": {
                        "_id": "$video_id",
                        "avg_embedding": {"$push": "$embedding"}
                    }}
                ]

            for doc in emb_collection.aggregate(emb_pipeline):
                embeddings = doc["avg_embedding"]
                if embeddings:
                    avg = np.mean(embeddings, axis=0).tolist()
                    video_embeddings[doc["_id"]] = avg
    except Exception as e:
        print(f"Error fetching embeddings for clustering: {e}")

    # Only cluster if we have embeddings
    clusters_response = []
    if video_embeddings:
        from clustering import cluster_videos, compute_2d_positions, auto_name_cluster
        name_lookup = {v.get("video_id", ""): v.get("name", "") for v in videos}

        if len(video_embeddings) >= 2:
            clusters = cluster_videos(video_embeddings)
            positions = compute_2d_positions(clusters)

            for i, c in enumerate(clusters):
                c["position"] = positions.get(c["id"], {"x": 0.5, "y": 0.5})
                auto = auto_name_cluster([name_lookup.get(vid, vid) for vid in c["video_ids"]])
                c["name"] = auto if auto else f"Cluster {i + 1}"

            clusters_response = clusters
        else:
            vid_id = list(video_embeddings.keys())[0]
            clusters_response = [{
                "id": "cluster_0",
                "video_ids": [vid_id],
                "centroid": video_embeddings[vid_id],
                "avg_similarity": 1.0,
                "position": {"x": 0.5, "y": 0.5},
                "name": name_lookup.get(vid_id, vid_id)
            }]

    return {"videos": videos, "clusters": clusters_response}


class ClusterSearchRequest(BaseModel):
    query: str
    centroids: dict[str, list]  # {cluster_id: 512d_centroid}


@app.post("/api/clusters/search")
async def search_clusters(req: ClusterSearchRequest):
    """Score clusters by semantic similarity to a text query."""
    client = get_search_client()
    try:
        result = client.bedrock.get_text_query_embedding(req.query)
        query_embedding = result["embedding"]
    except Exception as e:
        print(f"Cluster search embedding error: {e}")
        return {"scores": {}}

    from search_client import cosine_similarity
    scores = {}
    for cluster_id, centroid in req.centroids.items():
        scores[cluster_id] = round(cosine_similarity(query_embedding, centroid), 4)

    return {"scores": scores}


class DeleteVideosRequest(BaseModel):
    video_ids: list[str]
    backend: Optional[str] = None  # "mongodb", "s3vectors", or None for all
    index_mode: Optional[str] = None  # "unified", "multi", or None for all


@app.post("/api/videos/delete")
async def delete_videos(request: DeleteVideosRequest):
    """Delete video embeddings. Scoped to backend/index_mode if provided, otherwise deletes from all."""
    import boto3

    client = get_search_client()
    s3_client = boto3.client("s3")
    results = {}
    delete_all = request.backend is None

    for video_id in request.video_ids:
        vid_result = {}

        # Look up s3_uri from MongoDB BEFORE deleting (need it for proxy cleanup)
        proxy_key = None
        try:
            mongo_client = client.get_mongodb_client()
            doc = mongo_client.collection.find_one(
                {"video_id": video_id}, {"s3_uri": 1}
            )
            if doc and doc.get("s3_uri"):
                s3_uri = _normalize_s3_uri(doc["s3_uri"])
                parsed = urlparse(s3_uri)
                key = parsed.path.lstrip("/")
                if key.startswith("input/"):
                    proxy_key = key.replace("input/", "proxies/", 1)
                elif "/proxies/" in key or key.startswith("proxies/"):
                    proxy_key = key
                elif "WBD_project/Videos/proxy/" in key:
                    proxy_key = "proxies/" + key.split("WBD_project/Videos/proxy/", 1)[1]
                else:
                    proxy_key = key
        except Exception as e:
            print(f"Delete: s3_uri lookup failed for {video_id}: {e}")

        # Delete from S3 Vectors (scoped or all)
        if delete_all or request.backend == "s3vectors":
            try:
                s3v_client = client.get_s3_vectors_client()
                if delete_all or request.index_mode is None:
                    vid_result["s3_vectors"] = s3v_client.delete_video_embeddings(video_id)
                elif request.index_mode == "unified":
                    s3v_client.delete_from_index(video_id, "unified-embeddings")
                    vid_result["s3_vectors"] = {"unified": "deleted"}
                elif request.index_mode == "multi":
                    for idx in ["visual-embeddings", "audio-embeddings", "transcription-embeddings"]:
                        s3v_client.delete_from_index(video_id, idx)
                    vid_result["s3_vectors"] = {"multi": "deleted"}
            except Exception as e:
                vid_result["s3_vectors"] = {"error": str(e)}

        # Delete from MongoDB (scoped or all)
        if delete_all or request.backend == "mongodb":
            try:
                mongo_client = client.get_mongodb_client()
                if delete_all or request.index_mode is None:
                    vid_result["mongodb"] = mongo_client.delete_video_embeddings(video_id)
                elif request.index_mode == "unified":
                    r = mongo_client.db["unified-embeddings"].delete_many({"video_id": video_id})
                    vid_result["mongodb"] = {"unified": f"deleted {r.deleted_count}"}
                elif request.index_mode == "multi":
                    total = 0
                    for col in ["visual_embeddings", "audio_embeddings", "transcription_embeddings"]:
                        r = mongo_client.db[col].delete_many({"video_id": video_id})
                        total += r.deleted_count
                    vid_result["mongodb"] = {"multi": f"deleted {total}"}
            except Exception as e:
                vid_result["mongodb"] = {"error": str(e)}

        # Only delete proxy, S3 embeddings, fingerprint if deleting from ALL
        # (don't remove shared resources when just removing from one index)
        if delete_all:
            # Delete proxy video file from S3
            if proxy_key:
                try:
                    s3_client.delete_object(Bucket=S3_BUCKET, Key=proxy_key)
                    vid_result["s3_proxy"] = f"deleted {proxy_key}"
                except Exception as e:
                    vid_result["s3_proxy"] = f"error: {str(e)}"

            # Delete embedding output folder from S3
            try:
                paginator = s3_client.get_paginator("list_objects_v2")
                deleted_keys = []
                for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"embeddings/{video_id}"):
                    for obj in page.get("Contents", []):
                        deleted_keys.append(obj["Key"])
                if deleted_keys:
                    for i in range(0, len(deleted_keys), 1000):
                        batch = deleted_keys[i:i + 1000]
                        s3_client.delete_objects(
                            Bucket=S3_BUCKET,
                            Delete={"Objects": [{"Key": k} for k in batch]}
                        )
                    vid_result["s3_embeddings"] = f"deleted {len(deleted_keys)} objects"
            except Exception as e:
                vid_result["s3_embeddings"] = f"error: {str(e)}"

            # Delete fingerprint and thumbnail
            try:
                fp = mongo_client.get_video_fingerprint(video_id)
                if fp and fp.get("thumbnail_key"):
                    try:
                        s3_client.delete_object(Bucket=S3_BUCKET, Key=fp["thumbnail_key"])
                    except Exception:
                        pass
                mongo_client.delete_video_fingerprint(video_id)
                vid_result["fingerprint"] = "deleted"
            except Exception as e:
                vid_result["fingerprint"] = f"error: {str(e)}"

        print(f"Delete result for {video_id} (backend={request.backend}, mode={request.index_mode}): {vid_result}")
        results[video_id] = vid_result

    return {"deleted": len(request.video_ids), "results": results}


@app.get("/api/thumbnail/{video_id}/{segment_id}")
async def get_thumbnail(video_id: str, segment_id: int):
    """
    Get thumbnail for a video segment.
    Returns proxy video URL with time parameter for client-side thumbnail generation.
    """
    # Look up the segment to get start_time and video path
    client = get_search_client()
    segment = client.get_segment(video_id, segment_id)

    if not segment:
        return {"url": f"https://via.placeholder.com/320x180?text=Not+Found"}

    # Get video path and convert to proxy URL
    s3_uri = segment.get("s3_uri", "")
    parsed = urlparse(s3_uri)
    key = parsed.path.lstrip("/")

    # Map s3_uri key to proxies/ path for CloudFront delivery
    if "/proxies/" in key or key.startswith("proxies/"):
        proxy_key = key
    elif "WBD_project/Videos/proxy/" in key:
        proxy_key = "proxies/" + key.split("WBD_project/Videos/proxy/", 1)[1]
    elif key.startswith("input/"):
        proxy_key = key.replace("input/", "proxies/", 1)
    else:
        proxy_key = key
    proxy_url = f"https://{CLOUDFRONT_DOMAIN}/{proxy_key}"

    start_time = segment.get("start_time", 0)

    return {
        "url": proxy_url,
        "time": start_time + 3,  # Middle of 6-second segment
        "proxy": True
    }


@app.get("/api/video-url")
async def get_video_url(s3_uri: str = Query(..., description="S3 URI")):
    """Get CloudFront URL for a video."""
    parsed = urlparse(s3_uri)
    key = parsed.path.lstrip("/")
    return {"url": f"https://{CLOUDFRONT_DOMAIN}/{key}"}


class AnalyzeRequest(BaseModel):
    """Pegasus video analysis request."""
    s3_uri: str
    prompt: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@app.post("/api/analyze")
async def analyze_video(request: AnalyzeRequest):
    """
    Analyze a video segment using TwelveLabs Pegasus.

    - **s3_uri**: S3 URI of the video
    - **prompt**: Natural language question about the video
    - **start_time**: Optional segment start time (seconds)
    - **end_time**: Optional segment end time (seconds)
    """
    try:
        client = get_search_client()
        response_text = client.bedrock.analyze_video(
            s3_uri=request.s3_uri,
            prompt=request.prompt,
            start_time=request.start_time,
            end_time=request.end_time
        )
        return {"response": response_text}
    except Exception as e:
        print(f"Pegasus analysis error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


class ChatRequest(BaseModel):
    """Agent chat request."""
    message: str
    chat_history: Optional[list] = []
    context: Optional[dict] = None


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Agent-powered chat endpoint for the Analyze page.

    The agent reasons about user intent and calls appropriate tools:
    - search_segments: Marengo semantic search at segment level
    - search_assets: Aggregate search at video level
    - analyze_video: Pegasus video understanding
    """
    try:
        agent = get_chat_agent()
        context = request.context or {
            "settings": {
                "fusion_method": "dynamic",
                "backend": "s3vectors",
                "use_multi_index": True,
                "use_decomposition": False
            }
        }
        return agent.run(
            message=request.message,
            chat_history=request.chat_history or [],
            context=context
        )
    except Exception as e:
        print(f"Chat agent error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {"message": f"Error: {str(e)}", "actions": [], "tool_calls": []}


@app.post("/api/compare/find-similar")
async def compare_find_similar(request: Request):
    """Find videos similar to a reference video.
    Computes fingerprints on-the-fly from embeddings."""
    body = await request.json()
    video_id = body.get("video_id")
    backend = body.get("backend", "mongodb")
    index_mode = body.get("index_mode", "multi")
    if not video_id:
        return JSONResponse({"error": "video_id required"}, status_code=400)

    search_client = get_search_client()
    mongodb = search_client.get_mongodb_client()

    # Pick the right MongoDB collection based on index_mode
    if index_mode == "multi":
        collection = mongodb.db["visual_embeddings"]
    else:
        collection = mongodb.db[mongodb.COLLECTION_NAME]

    # For S3 Vectors backend, we still need MongoDB for embeddings data (S3 Vectors
    # doesn't support full doc retrieval). But we use the S3 Vectors video list.
    if backend == "s3vectors":
        try:
            s3v = search_client.get_s3_vectors_client()
            idx = s3v.UNIFIED_INDEX_NAME if index_mode == "unified" else s3v.INDEX_NAMES.get("visual", "visual-embeddings")
            s3v_videos = s3v.list_videos(index_name=idx)
            s3v_ids = {v["video_id"] for v in s3v_videos}
        except Exception as e:
            print(f"S3 Vectors list error: {e}")
            s3v_ids = set()

        if video_id not in s3v_ids:
            return JSONResponse({"error": "Video not found in this index. Try a different backend or index mode."}, status_code=404)
        video_ids = list(s3v_ids)
    else:
        video_ids = collection.distinct("video_id")
        if video_id not in video_ids:
            return JSONResponse({"error": "Video not found in embeddings. Upload and index the video first."}, status_code=404)

    # Helper to resolve video CloudFront URL from s3_uri
    def _resolve_video_url(vid_id):
        # Try unified collection first (has the most data)
        for coll_name in [mongodb.COLLECTION_NAME, "visual_embeddings"]:
            seg = mongodb.db[coll_name].find_one({"video_id": vid_id}, {"s3_uri": 1, "_id": 0})
            if seg and seg.get("s3_uri"):
                s3_uri = _normalize_s3_uri(seg["s3_uri"])
                parsed = urlparse(s3_uri)
                key = parsed.path.lstrip("/")
                if key.startswith("input/"):
                    key = key.replace("input/", "proxies/", 1)
                return f"https://{CLOUDFRONT_DOMAIN}/{key}"
        return None

    if len(video_ids) < 2:
        return {"reference": {"video_id": video_id, "name": video_id, "video_url": _resolve_video_url(video_id)}, "results": []}

    # Compute fingerprints on-the-fly from MongoDB embeddings
    # Use unified collection when available (has all modalities in one place)
    fp_collection = mongodb.db[mongodb.COLLECTION_NAME]
    all_fps = []
    segments_by_video = {}  # Keep segments for segment-level re-scoring
    for vid in video_ids:
        segments = list(fp_collection.find(
            {"video_id": vid},
            {"modality_type": 1, "embedding": 1, "segment_id": 1, "start_time": 1, "end_time": 1, "_id": 0}
        ))
        if not segments:
            # Fallback: try multi-index collections
            for coll_name in ["visual_embeddings", "audio_embeddings", "transcription_embeddings"]:
                modality = coll_name.replace("_embeddings", "")
                for doc in mongodb.db[coll_name].find({"video_id": vid}, {"embedding": 1, "segment_id": 1, "start_time": 1, "end_time": 1, "_id": 0}):
                    doc["modality_type"] = modality
                    segments.append(doc)
        if not segments:
            continue
        segments_by_video[vid] = segments
        fp = compute_fingerprint(segments)
        fp["video_id"] = vid
        # Derive name from s3_uri (try unified, then multi-index)
        seg = fp_collection.find_one({"video_id": vid}, {"s3_uri": 1, "_id": 0})
        if not seg or not seg.get("s3_uri"):
            for coll_name in ["visual_embeddings", "audio_embeddings", "transcription_embeddings"]:
                seg = mongodb.db[coll_name].find_one({"video_id": vid}, {"s3_uri": 1, "_id": 0})
                if seg and seg.get("s3_uri"):
                    break
        if seg and seg.get("s3_uri"):
            filename = os.path.basename(seg["s3_uri"])
            fp["video_name"] = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
        else:
            fp["video_name"] = vid
        all_fps.append(fp)

    results = find_similar_videos(video_id, all_fps)

    # Re-score top results using segment-level alignment for accuracy
    # (fingerprint-based scores can inflate due to mean embedding smoothing)
    ref_segs = segments_by_video.get(video_id)
    if ref_segs:
        for r in results:
            cmp_segs = segments_by_video.get(r["video_id"])
            if cmp_segs:
                diff = align_segments(ref_segs, cmp_segs)
                r["overall_similarity"] = diff["summary"]["overall_similarity"]
                r["modality_scores"] = diff["modality_similarity"]
        # Re-sort after re-scoring
        results.sort(key=lambda x: x["overall_similarity"], reverse=True)

    # Add video URLs to results
    for r in results:
        url = _resolve_video_url(r["video_id"])
        if url:
            r["video_url"] = url

    # Reference info
    ref_fp = next((fp for fp in all_fps if fp["video_id"] == video_id), {})

    return {
        "reference": {
            "video_id": video_id,
            "name": ref_fp.get("video_name", video_id),
            "segment_count": ref_fp.get("segment_count", 0),
            "duration": ref_fp.get("total_duration", 0.0),
            "video_url": _resolve_video_url(video_id),
        },
        "results": results,
    }


@app.post("/api/compare/diff")
async def compare_diff(request: Request):
    """Compute segment-level diff between two videos."""
    body = await request.json()
    ref_id = body.get("reference_video_id")
    cmp_id = body.get("compare_video_id")
    if not ref_id or not cmp_id:
        return JSONResponse({"error": "reference_video_id and compare_video_id required"}, status_code=400)

    search_client = get_search_client()
    mongodb = search_client.get_mongodb_client()

    ref_segments = mongodb.get_segments_for_video(ref_id)
    cmp_segments = mongodb.get_segments_for_video(cmp_id)

    if not ref_segments:
        return JSONResponse({"error": f"No segments found for reference video {ref_id}"}, status_code=404)
    if not cmp_segments:
        return JSONResponse({"error": f"No segments found for compare video {cmp_id}"}, status_code=404)

    result = align_segments(ref_segments, cmp_segments)

    # Include technical metadata from fingerprints (with on-the-fly fallback)
    ref_fp = mongodb.get_video_fingerprint(ref_id)
    cmp_fp = mongodb.get_video_fingerprint(cmp_id)
    result["reference_metadata"] = _get_tech_metadata(ref_id, ref_fp)
    result["compare_metadata"] = _get_tech_metadata(cmp_id, cmp_fp)

    return result


@app.post("/api/compare/multi-diff")
async def compare_multi_diff(request: Request):
    """Compare a reference video against multiple comparison videos simultaneously."""
    body = await request.json()
    reference_id = body.get("reference_id")
    compare_ids = body.get("compare_ids", [])

    if not reference_id or not compare_ids:
        return JSONResponse({"error": "reference_id and compare_ids required"}, status_code=400)

    if len(compare_ids) > 8:
        return JSONResponse({"error": "Maximum 8 comparison videos"}, status_code=400)

    # Get clients (same pattern as existing /api/compare/diff endpoint)
    search_client = get_search_client()
    mongodb = search_client.get_mongodb_client()

    # Fetch reference data
    ref_segments = mongodb.get_segments_for_video(reference_id)
    if not ref_segments:
        return JSONResponse({"error": f"No segments found for reference video {reference_id}"}, status_code=404)

    ref_fp = mongodb.get_video_fingerprint(reference_id)
    ref_metadata = {
        "name": ref_fp.get("video_name", reference_id) if ref_fp else reference_id,
        "duration": ref_fp.get("total_duration", 0) if ref_fp else 0,
        "segment_count": ref_fp.get("segment_count", 0) if ref_fp else 0,
        "s3_key": ref_fp.get("s3_key", "") if ref_fp else "",
    }
    ref_tech = _get_tech_metadata(reference_id, ref_fp)

    # Process each comparison video
    comparisons = []
    for cmp_id in compare_ids:
        try:
            cmp_segments = mongodb.get_segments_for_video(cmp_id)
            cmp_fp = mongodb.get_video_fingerprint(cmp_id)

            alignment = align_segments(ref_segments, cmp_segments)

            cmp_metadata = {
                "name": cmp_fp.get("video_name", cmp_id) if cmp_fp else cmp_id,
                "duration": cmp_fp.get("total_duration", 0) if cmp_fp else 0,
                "segment_count": cmp_fp.get("segment_count", 0) if cmp_fp else 0,
                "s3_key": cmp_fp.get("s3_key", "") if cmp_fp else "",
            }
            cmp_tech = _get_tech_metadata(cmp_id, cmp_fp)

            comparisons.append({
                "video_id": cmp_id,
                "metadata": cmp_metadata,
                "technical_metadata": cmp_tech,
                "alignment": alignment
            })
        except Exception as e:
            comparisons.append({
                "video_id": cmp_id,
                "error": str(e),
                "metadata": {"name": cmp_id},
                "technical_metadata": None,
                "alignment": None
            })

    # Resolve video URLs for all videos
    video_urls = {}
    all_ids = [reference_id] + compare_ids
    for vid_id in all_ids:
        try:
            s3_key = _resolve_s3_key(vid_id)
            if s3_key and CLOUDFRONT_DOMAIN:
                video_urls[vid_id] = f"https://{CLOUDFRONT_DOMAIN}/{s3_key}"
        except Exception:
            pass

    return {
        "reference": {
            "video_id": reference_id,
            "metadata": ref_metadata,
            "technical_metadata": ref_tech
        },
        "comparisons": comparisons,
        "video_urls": video_urls
    }


@app.get("/api/compare/report/{reference_id}/{compare_id}")
async def compare_report(reference_id: str, compare_id: str):
    """Generate full comparison report with video metadata."""
    search_client = get_search_client()
    mongodb = search_client.get_mongodb_client()

    ref_fp = mongodb.get_video_fingerprint(reference_id)
    cmp_fp = mongodb.get_video_fingerprint(compare_id)
    ref_segments = mongodb.get_segments_for_video(reference_id)
    cmp_segments = mongodb.get_segments_for_video(compare_id)

    if not ref_segments or not cmp_segments:
        return JSONResponse({"error": "Segments not found for one or both videos"}, status_code=404)

    diff = align_segments(ref_segments, cmp_segments)

    return {
        "reference": {
            "video_id": reference_id,
            "name": ref_fp.get("video_name", reference_id) if ref_fp else reference_id,
            "segment_count": ref_fp.get("segment_count", 0) if ref_fp else len(set(s["segment_id"] for s in ref_segments)),
            "duration": ref_fp.get("total_duration", 0.0) if ref_fp else 0.0,
            "technical_metadata": _get_tech_metadata(reference_id, ref_fp),
        },
        "compare": {
            "video_id": compare_id,
            "name": cmp_fp.get("video_name", compare_id) if cmp_fp else compare_id,
            "segment_count": cmp_fp.get("segment_count", 0) if cmp_fp else len(set(s["segment_id"] for s in cmp_segments)),
            "duration": cmp_fp.get("total_duration", 0.0) if cmp_fp else 0.0,
            "technical_metadata": _get_tech_metadata(compare_id, cmp_fp),
        },
        **diff,
    }


@app.get("/api/compare/report/{reference_id}/{compare_id}/csv")
async def compare_report_csv(reference_id: str, compare_id: str):
    """Export comparison report as CSV."""
    import csv
    import io

    search_client = get_search_client()
    mongodb = search_client.get_mongodb_client()
    ref_segments = mongodb.get_segments_for_video(reference_id)
    cmp_segments = mongodb.get_segments_for_video(compare_id)

    if not ref_segments or not cmp_segments:
        return JSONResponse({"error": "Segments not found"}, status_code=404)

    diff = align_segments(ref_segments, cmp_segments)

    # Get technical metadata for header
    ref_fp = mongodb.get_video_fingerprint(reference_id)
    cmp_fp = mongodb.get_video_fingerprint(compare_id)

    output = io.StringIO()
    writer = csv.writer(output)

    # Technical metadata header rows
    ref_meta = (ref_fp or {}).get("technical_metadata")
    cmp_meta = (cmp_fp or {}).get("technical_metadata")
    if ref_meta or cmp_meta:
        def _meta_summary(meta):
            if not meta:
                return "N/A"
            v = meta.get("video", {})
            a = meta.get("audio", {})
            c = meta.get("container", {})
            parts = []
            if v.get("codec"):
                parts.append(f"{v['codec']}{' '+v['profile'] if v.get('profile') else ''}")
            if v.get("width"):
                parts.append(f"{v['width']}x{v['height']}")
            if v.get("framerate"):
                parts.append(f"{v['framerate']}fps")
            if c.get("bitrate_kbps"):
                parts.append(f"{c['bitrate_kbps']/1000:.1f}Mbps")
            if a.get("codec"):
                parts.append(f"{a['codec']} {a.get('channel_layout', '')}")
            if c.get("file_size_mb"):
                parts.append(f"{c['file_size_mb']}MB")
            return " | ".join(parts)
        writer.writerow(["# Reference", _meta_summary(ref_meta)])
        writer.writerow(["# Compare", _meta_summary(cmp_meta)])
        writer.writerow([])

    writer.writerow(["status", "similarity", "ref_segment_id", "ref_start", "ref_end",
                     "cmp_segment_id", "cmp_start", "cmp_end",
                     "visual_score", "audio_score", "transcription_score"])

    for seg in diff["segments"]:
        ref = seg.get("reference") or {}
        cmp = seg.get("compare") or {}
        scores = seg.get("modality_scores") or {}
        writer.writerow([
            seg["status"], seg.get("similarity", ""),
            ref.get("segment_id", ""), ref.get("start_time", ""), ref.get("end_time", ""),
            cmp.get("segment_id", ""), cmp.get("start_time", ""), cmp.get("end_time", ""),
            scores.get("visual", ""), scores.get("audio", ""), scores.get("transcription", ""),
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=compare_{reference_id}_vs_{compare_id}.csv"}
    )


@app.post("/api/compare/export-segments")
async def compare_export_segments(request: Request):
    """Export non-matching segments as a JSON manifest with pre-signed URLs."""
    body = await request.json()
    ref_id = body.get("reference_video_id")
    cmp_id = body.get("compare_video_id")
    threshold = body.get("threshold", 1.0)
    include_statuses = body.get("include_statuses", ["changed", "missing", "added"])

    if not ref_id or not cmp_id:
        return JSONResponse({"error": "reference_video_id and compare_video_id required"}, status_code=400)

    search_client = get_search_client()
    mongodb = search_client.get_mongodb_client()

    ref_segments = mongodb.get_segments_for_video(ref_id)
    cmp_segments = mongodb.get_segments_for_video(cmp_id)

    if not ref_segments or not cmp_segments:
        return JSONResponse({"error": "Segments not found for one or both videos"}, status_code=404)

    diff = align_segments(ref_segments, cmp_segments)

    # Resolve video URLs
    def _resolve_url(vid_id):
        for coll_name in [mongodb.COLLECTION_NAME, "visual_embeddings"]:
            seg = mongodb.db[coll_name].find_one({"video_id": vid_id}, {"s3_uri": 1, "_id": 0})
            if seg and seg.get("s3_uri"):
                s3_uri = _normalize_s3_uri(seg["s3_uri"])
                parsed = urlparse(s3_uri)
                key = parsed.path.lstrip("/")
                if key.startswith("input/"):
                    key = key.replace("input/", "proxies/", 1)
                return f"https://{CLOUDFRONT_DOMAIN}/{key}"
        return None

    ref_url = _resolve_url(ref_id)
    cmp_url = _resolve_url(cmp_id)

    # Filter segments below threshold or matching statuses
    exported = []
    for seg in diff["segments"]:
        status = seg["status"]
        sim = seg.get("similarity")

        # Include if status matches OR similarity below threshold
        if status in include_statuses or (sim is not None and sim < threshold):
            entry = {
                "status": status,
                "similarity": sim,
                "modality_scores": seg.get("modality_scores"),
            }
            if seg.get("reference"):
                entry["reference"] = {
                    **seg["reference"],
                    "video_url": ref_url,
                }
            if seg.get("compare"):
                entry["compare"] = {
                    **seg["compare"],
                    "video_url": cmp_url,
                }
            exported.append(entry)

    # Get fingerprint metadata
    ref_fp = mongodb.get_video_fingerprint(ref_id)
    cmp_fp = mongodb.get_video_fingerprint(cmp_id)

    return {
        "export": {
            "reference_video_id": ref_id,
            "reference_name": ref_fp.get("video_name", ref_id) if ref_fp else ref_id,
            "compare_video_id": cmp_id,
            "compare_name": cmp_fp.get("video_name", cmp_id) if cmp_fp else cmp_id,
            "threshold": threshold,
            "include_statuses": include_statuses,
            "segment_count": len(exported),
            "total_segments": len(diff["segments"]),
            "overall_similarity": diff["summary"]["overall_similarity"],
            "segments": exported,
        }
    }


def _get_ffmpeg_path():
    """Find ffmpeg binary — App Runner build or system PATH."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "ffmpeg"),  # /app/ffmpeg from build
        "/app/ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    # Fallback: try PATH
    import shutil
    return shutil.which("ffmpeg")


def _extract_tech_metadata(video_id: str, s3_key: str = None) -> dict:
    """Extract technical metadata from the ORIGINAL source file in S3.

    Tries originals/ path first (true source specs), falls back to proxies/.
    Also backfills the fingerprint in MongoDB for future calls.
    """
    import re as _re
    import subprocess
    import tempfile

    ffmpeg_bin = _get_ffmpeg_path()
    if not ffmpeg_bin:
        return None

    if not s3_key:
        s3_key = _resolve_s3_key(video_id)
    if not s3_key:
        return None

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    bucket = os.environ.get("S3_BUCKET", "multi-modal-video-search-app")

    # Try original file first (originals/ has the untouched source)
    original_key = s3_key.replace("proxies/", "originals/", 1) if "proxies/" in s3_key else None
    download_key = s3_key  # default to proxy

    if original_key:
        try:
            s3.head_object(Bucket=bucket, Key=original_key)
            download_key = original_key
        except Exception:
            pass  # original not found, use proxy

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        s3.download_file(bucket, download_key, tmp.name)

        probe = subprocess.run(
            [ffmpeg_bin, "-i", tmp.name],
            capture_output=True, text=True, timeout=15
        )
        probe_output = probe.stderr or ""
        file_size = os.path.getsize(tmp.name)
        os.unlink(tmp.name)

        # Parse metadata (same logic as Lambda _parse_technical_metadata)
        is_original = download_key != s3_key
        metadata = {"container": {}, "video": {}, "audio": {}, "source": {"s3_key": download_key, "is_original": is_original}}

        fmt_match = _re.search(r'Input #\d+,\s*([^,]+(?:,[^,]+)*),\s*from', probe_output)
        if fmt_match:
            metadata["container"]["format"] = fmt_match.group(1).strip()

        dur_match = _re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', probe_output)
        if dur_match:
            metadata["container"]["duration"] = round(
                int(dur_match.group(1)) * 3600 + int(dur_match.group(2)) * 60 + float(dur_match.group(3)), 2
            )

        br_match = _re.search(r'bitrate:\s*(\d+)\s*kb/s', probe_output)
        if br_match:
            metadata["container"]["bitrate_kbps"] = int(br_match.group(1))

        if file_size > 0:
            metadata["container"]["file_size_mb"] = round(file_size / (1024 * 1024), 2)

        video_match = _re.search(r'Stream #\d+:\d+[^:]*:\s*Video:\s*(.+)', probe_output)
        if video_match:
            vline = video_match.group(1)
            codec_match = _re.match(r'(\w+)(?:\s*\(([^)]+)\))?', vline)
            if codec_match:
                metadata["video"]["codec"] = codec_match.group(1)
                if codec_match.group(2):
                    metadata["video"]["profile"] = codec_match.group(2).strip()
            cs_match = _re.search(r',\s*(yuv\w+|rgb\w+|gbr\w+|gray\w*|nv\d+)', vline)
            if cs_match:
                cs = cs_match.group(1)
                metadata["video"]["color_space"] = cs
                metadata["video"]["bit_depth"] = 10 if "10le" in cs or "10be" in cs else (12 if "12le" in cs or "12be" in cs else 8)
            res_match = _re.search(r'(\d{2,5})x(\d{2,5})', vline)
            if res_match:
                metadata["video"]["width"] = int(res_match.group(1))
                metadata["video"]["height"] = int(res_match.group(2))
            dar_match = _re.search(r'DAR\s+(\d+:\d+)', vline)
            if dar_match:
                metadata["video"]["display_aspect_ratio"] = dar_match.group(1)
            fps_match = _re.search(r'(\d+(?:\.\d+)?)\s*(?:fps|tbr)', vline)
            if fps_match:
                metadata["video"]["framerate"] = float(fps_match.group(1))
            vbr_match = _re.search(r'(\d+)\s*kb/s', vline)
            if vbr_match:
                metadata["video"]["bitrate_kbps"] = int(vbr_match.group(1))
            if _re.search(r'\b(tff|bff|interlaced)\b', vline, _re.IGNORECASE):
                metadata["video"]["scan_type"] = "interlaced"
            else:
                metadata["video"]["scan_type"] = "progressive"

        audio_match = _re.search(r'Stream #\d+:\d+[^:]*:\s*Audio:\s*(.+)', probe_output)
        if audio_match:
            aline = audio_match.group(1)
            ac_match = _re.match(r'(\w+)', aline)
            if ac_match:
                metadata["audio"]["codec"] = ac_match.group(1)
            sr_match = _re.search(r'(\d+)\s*Hz', aline)
            if sr_match:
                metadata["audio"]["sample_rate"] = int(sr_match.group(1))
            ch_match = _re.search(r'(mono|stereo|5\.1|7\.1|(\d+)\s*channels)', aline, _re.IGNORECASE)
            if ch_match:
                metadata["audio"]["channel_layout"] = ch_match.group(1)
            abr_match = _re.search(r'(\d+)\s*kb/s', aline)
            if abr_match:
                metadata["audio"]["bitrate_kbps"] = int(abr_match.group(1))

        # Backfill into MongoDB fingerprint
        try:
            search_client = get_search_client()
            mongodb = search_client.get_mongodb_client()
            mongodb.db["video_fingerprints"].update_one(
                {"video_id": video_id},
                {"$set": {"technical_metadata": metadata}},
            )
            logging.getLogger(__name__).info(f"Backfilled technical_metadata for {video_id}")
        except Exception:
            pass

        return metadata
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to extract tech metadata for {video_id}: {e}")
        return None


def _get_tech_metadata(video_id: str, fingerprint: dict) -> dict:
    """Get technical metadata from fingerprint, falling back to on-the-fly extraction.

    Re-extracts if stored metadata came from proxy (missing is_original flag).
    """
    stored = fingerprint.get("technical_metadata") if fingerprint else None
    if stored and stored.get("source", {}).get("is_original"):
        return stored
    # Missing or from proxy — extract from original
    return _extract_tech_metadata(video_id)


def _resolve_s3_key(vid_id):
    """Resolve video_id to S3 proxy key."""
    search_client = get_search_client()
    mongodb = search_client.get_mongodb_client()
    for coll_name in [mongodb.COLLECTION_NAME, "visual_embeddings"]:
        seg = mongodb.db[coll_name].find_one({"video_id": vid_id}, {"s3_uri": 1, "_id": 0})
        if seg and seg.get("s3_uri"):
            s3_uri = _normalize_s3_uri(seg["s3_uri"])
            parsed = urlparse(s3_uri)
            key = parsed.path.lstrip("/")
            if key.startswith("input/"):
                key = key.replace("input/", "proxies/", 1)
            return key
    return None


@app.post("/api/compare/analyze-segment")
async def compare_analyze_segment(request: Request):
    """Analyze visual differences in a segment pair using Claude Haiku.

    Extracts frames from both videos at the segment timestamps,
    sends pairs to Claude for visual comparison, returns per-frame analysis.
    """
    import base64
    import subprocess
    import tempfile

    body = await request.json()
    ref_id = body.get("reference_video_id")
    cmp_id = body.get("compare_video_id")
    ref_start = body.get("ref_start", 0)
    ref_end = body.get("ref_end", 1)
    cmp_start = body.get("cmp_start", 0)
    cmp_end = body.get("cmp_end", 1)

    if not ref_id or not cmp_id:
        return JSONResponse({"error": "reference_video_id and compare_video_id required"}, status_code=400)

    ffmpeg_bin = _get_ffmpeg_path()
    if not ffmpeg_bin:
        return JSONResponse({"error": "ffmpeg not available on server"}, status_code=500)

    # Resolve S3 keys for both videos
    ref_key = _resolve_s3_key(ref_id)
    cmp_key = _resolve_s3_key(cmp_id)
    if not ref_key or not cmp_key:
        return JSONResponse({"error": "Could not resolve video files"}, status_code=404)

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    bucket = os.environ.get("S3_BUCKET", "multi-modal-video-search-app")

    try:
        # Download both videos to temp files
        ref_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        cmp_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

        s3.download_file(bucket, ref_key, ref_tmp.name)
        s3.download_file(bucket, cmp_key, cmp_tmp.name)

        # Extract frames matching source framerate (capped at 30 for cost/size)
        segment_duration = max(ref_end - ref_start, cmp_end - cmp_start)
        # Detect framerate from reference video
        try:
            probe_result = subprocess.run(
                [ffmpeg_bin.replace("ffmpeg", "ffprobe"), "-v", "error",
                 "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                 "-of", "csv=p=0", ref_tmp.name],
                capture_output=True, text=True, timeout=10
            )
            fps_str = probe_result.stdout.strip()
            if "/" in fps_str:
                num, den = fps_str.split("/")
                source_fps = float(num) / float(den) if float(den) > 0 else 24
            else:
                source_fps = float(fps_str) if fps_str else 24
        except Exception:
            source_fps = 24
        num_frames = max(2, min(30, int(segment_duration * source_fps)))  # match fps, cap at 30

        def extract_frames(video_path, start, end, n_frames):
            """Extract n_frames as JPEG bytes from video between start-end."""
            duration = end - start
            if duration <= 0:
                duration = 1
            fps = n_frames / duration
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.run([
                    ffmpeg_bin, "-y",
                    "-ss", str(start), "-t", str(duration),
                    "-i", video_path,
                    "-vf", f"fps={fps},scale=640:-1",
                    "-q:v", "4",
                    os.path.join(tmpdir, "frame_%03d.jpg")
                ], capture_output=True, timeout=30)

                frames = []
                for f in sorted(os.listdir(tmpdir)):
                    if f.endswith(".jpg"):
                        with open(os.path.join(tmpdir, f), "rb") as fh:
                            frames.append(fh.read())
                return frames

        ref_frames = extract_frames(ref_tmp.name, ref_start, ref_end, num_frames)
        cmp_frames = extract_frames(cmp_tmp.name, cmp_start, cmp_end, num_frames)

        # Clean up video files
        os.unlink(ref_tmp.name)
        os.unlink(cmp_tmp.name)

        if not ref_frames or not cmp_frames:
            return JSONResponse({"error": "Could not extract frames from videos"}, status_code=500)

        # Pair frames (use min length)
        n_pairs = min(len(ref_frames), len(cmp_frames))
        ref_frames = ref_frames[:n_pairs]
        cmp_frames = cmp_frames[:n_pairs]

        # Build Claude Haiku prompt with frame pairs
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

        content_blocks = [
            {"text": (
                f"You are a broadcast QC analyst comparing two video segments frame-by-frame.\n\n"
                f"Reference segment: {ref_start:.1f}s - {ref_end:.1f}s\n"
                f"Compare segment: {cmp_start:.1f}s - {cmp_end:.1f}s\n\n"
                f"I'm showing you {n_pairs} frame pairs. For each pair, the first image is the REFERENCE "
                f"and the second image is the COMPARE.\n\n"
                f"For each pair, determine if they are IDENTICAL or DIFFERENT. "
                f"If different, describe the specific visual difference concisely "
                f"(e.g., 'text overlay changed', 'color grading differs', 'logo added', 'frame cropped').\n\n"
                f"Respond as a JSON array with one object per pair:\n"
                f'[{{"frame": 1, "identical": true/false, "difference": "description or null"}}]\n\n'
                f"Output ONLY the JSON array, no other text."
            )}
        ]

        for i in range(n_pairs):
            frame_time = ref_start + (i / max(n_pairs - 1, 1)) * (ref_end - ref_start)
            content_blocks.append({"text": f"\n--- Frame pair {i+1} (t={frame_time:.2f}s) ---"})
            content_blocks.append({
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": ref_frames[i]}
                }
            })
            content_blocks.append({
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": cmp_frames[i]}
                }
            })

        response = bedrock.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[{"role": "user", "content": content_blocks}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.1}
        )

        # Parse response
        output_text = ""
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                output_text += block["text"]

        import re as re_mod
        # Extract JSON array from response
        json_match = re_mod.search(r'\[.*\]', output_text, re_mod.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            analysis = [{"frame": i+1, "identical": False, "difference": "Could not parse analysis"} for i in range(n_pairs)]

        # Build frame thumbnails as base64 for frontend display
        frame_data = []
        for i in range(n_pairs):
            frame_time = ref_start + (i / max(n_pairs - 1, 1)) * (ref_end - ref_start)
            entry = {
                "frame": i + 1,
                "timestamp": round(frame_time, 2),
                "ref_image": base64.b64encode(ref_frames[i]).decode(),
                "cmp_image": base64.b64encode(cmp_frames[i]).decode(),
                "identical": analysis[i].get("identical", True) if i < len(analysis) else True,
                "difference": analysis[i].get("difference") if i < len(analysis) else None,
            }
            frame_data.append(entry)

        # Count differences
        diff_count = sum(1 for f in frame_data if not f["identical"])

        return {
            "segment": {
                "ref_start": ref_start,
                "ref_end": ref_end,
                "cmp_start": cmp_start,
                "cmp_end": cmp_end,
            },
            "total_frames": n_pairs,
            "differences_found": diff_count,
            "frames": frame_data,
        }

    except Exception as e:
        logging.error(f"Segment analysis failed: {e}", exc_info=True)
        # Clean up temp files
        for tmp in [ref_tmp.name, cmp_tmp.name]:
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return JSONResponse({"error": f"Analysis failed: {str(e)}"}, status_code=500)


# =============================================
# Upload Endpoints
# =============================================

@app.post("/api/upload/presign")
async def upload_presign(request: Request):
    """Generate a presigned URL for direct browser-to-S3 upload."""
    body = await request.json()
    filename = body.get("filename")
    settings = body.get("settings", {})

    if not filename:
        return JSONResponse({"error": "filename required"}, status_code=400)

    allowed_ext = (".mp4", ".mov", ".mxf")
    if not filename.lower().endswith(tuple(allowed_ext)):
        return JSONResponse({"error": f"Unsupported file type. Allowed: {', '.join(allowed_ext)}"}, status_code=400)

    upload_id = str(uuid.uuid4())[:12]
    s3_key = f"input/{filename}"
    bucket = S3_BUCKET

    region = os.environ.get("AWS_REGION", "us-east-1")
    s3_client = boto3.client(
        "s3",
        region_name=region,
        config=botocore.config.Config(signature_version="s3v4"),
    )

    # Generate presigned PUT URL (valid 1 hour, up to 6GB)
    # Set correct content type so S3 stores it properly for browser playback
    ext = os.path.splitext(filename)[1].lower()
    content_type = {"mp4": "video/mp4", "mov": "video/quicktime", "mxf": "application/mxf"}.get(ext.lstrip("."), "application/octet-stream")
    presigned_url = s3_client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": s3_key, "ContentType": content_type},
        ExpiresIn=3600,
    )

    # Write initial status marker
    status_key = f"status/{upload_id}.json"
    s3_client.put_object(
        Bucket=bucket, Key=status_key,
        Body=json.dumps({"status": "uploading", "progress": 0, "message": "Uploading to S3...", "s3_key": s3_key}),
        ContentType="application/json"
    )

    return {
        "upload_id": upload_id,
        "presigned_url": presigned_url,
        "s3_key": s3_key,
        "content_type": content_type,
        "settings": settings,
    }


@app.post("/api/upload/start-processing")
async def upload_start_processing(request: Request):
    """Invoke Lambda to process an already-uploaded file."""
    body = await request.json()
    upload_id = body.get("upload_id")
    s3_key = body.get("s3_key")
    settings = body.get("settings", {})

    if not upload_id or not s3_key:
        return JSONResponse({"error": "upload_id and s3_key required"}, status_code=400)

    bucket = S3_BUCKET
    status_key = f"status/{upload_id}.json"
    s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    s3_client.put_object(
        Bucket=bucket, Key=status_key,
        Body=json.dumps({"status": "processing", "progress": 30, "message": "Upload complete. Starting embedding generation...", "s3_key": s3_key}),
        ContentType="application/json"
    )

    try:
        lambda_client = boto3.client("lambda", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        payload = {
            "s3_key": s3_key,
            "bucket": bucket,
            "segmentation_method": settings.get("segmentation", "fixed"),
            "min_duration_sec": settings.get("duration", 1),
            "segment_length_sec": settings.get("duration", 1),
            "embedding_types": settings.get("embedding_types", ["visual", "audio", "transcription"]),
            "storage_backends": settings.get("backends", ["mongodb"]),
            "index_modes": settings.get("index_modes", ["single"]),
            "upload_id": upload_id,
            "status_key": status_key,
        }
        lambda_client.invoke(
            FunctionName=os.environ.get("LAMBDA_FUNCTION_NAME", "video-search-processor"),
            InvocationType="Event",
            Payload=json.dumps(payload).encode(),
        )
        s3_client.put_object(
            Bucket=bucket, Key=status_key,
            Body=json.dumps({"status": "processing", "progress": 40, "message": "Lambda invoked. Generating embeddings...", "s3_key": s3_key}),
            ContentType="application/json"
        )
    except Exception as e:
        s3_client.put_object(
            Bucket=bucket, Key=status_key,
            Body=json.dumps({"status": "error", "progress": 30, "message": f"Lambda invocation failed: {e}", "s3_key": s3_key}),
            ContentType="application/json"
        )

    return {"upload_id": upload_id, "status": "processing"}


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    settings: str = Form(default="{}")
):
    """Legacy upload endpoint — streams through server. Use /api/upload/presign for large files."""
    allowed_ext = (".mp4", ".mov", ".mxf")
    if not file.filename.lower().endswith(allowed_ext):
        return JSONResponse(
            {"error": f"Unsupported file type. Allowed: {', '.join(allowed_ext)}"},
            status_code=400
        )

    upload_id = str(uuid.uuid4())[:12]
    settings_dict = json.loads(settings)

    bucket = S3_BUCKET
    s3_key = f"input/{file.filename}"
    status_key = f"status/{upload_id}.json"

    s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    def write_status(status, progress, message):
        s3_client.put_object(
            Bucket=bucket, Key=status_key,
            Body=json.dumps({"status": status, "progress": progress, "message": message, "s3_key": s3_key}),
            ContentType="application/json"
        )

    write_status("uploading", 0, "Uploading to S3...")

    try:
        s3_client.upload_fileobj(file.file, bucket, s3_key)
        write_status("processing", 30, "Upload complete. Starting embedding generation...")
    except Exception as e:
        write_status("error", 0, str(e))
        return JSONResponse({"upload_id": upload_id, "status": "error", "message": str(e)}, status_code=500)

    try:
        lambda_client = boto3.client("lambda", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        payload = {
            "s3_key": s3_key,
            "bucket": bucket,
            "segmentation_method": settings_dict.get("segmentation", "fixed"),
            "min_duration_sec": settings_dict.get("duration", 1),
            "segment_length_sec": settings_dict.get("duration", 1),
            "embedding_types": settings_dict.get("embedding_types", ["visual", "audio", "transcription"]),
            "storage_backends": settings_dict.get("backends", ["mongodb"]),
            "index_modes": settings_dict.get("index_modes", ["single"]),
            "upload_id": upload_id,
            "status_key": status_key,
        }
        lambda_client.invoke(
            FunctionName=os.environ.get("LAMBDA_FUNCTION_NAME", "video-search-processor"),
            InvocationType="Event",
            Payload=json.dumps(payload).encode(),
        )
        write_status("processing", 40, "Lambda invoked. Generating embeddings...")
    except Exception as e:
        write_status("error", 30, f"Lambda invocation failed: {e}")

    return {"upload_id": upload_id, "status": "processing"}


@app.get("/api/upload/{upload_id}/status")
async def upload_status(upload_id: str):
    """Poll upload processing status via S3 marker object.
    Also checks if Lambda finished by looking for embeddings in MongoDB/S3 Vectors."""
    bucket = S3_BUCKET
    status_key = f"status/{upload_id}.json"
    try:
        s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        obj = s3_client.get_object(Bucket=bucket, Key=status_key)
        status_data = json.loads(obj["Body"].read().decode())

        # If status is "processing" (stuck at Lambda stage), check if embeddings appeared
        if status_data.get("status") == "processing" and status_data.get("progress", 0) >= 40:
            s3_key = status_data.get("s3_key")  # e.g. input/filename.mp4
            if s3_key:
                try:
                    # Check MongoDB for embeddings (most reliable signal)
                    search_client = get_search_client()
                    mongo = search_client.get_mongodb_client()
                    filename = s3_key.replace("input/", "")
                    doc = mongo.collection.find_one({"s3_uri": {"$regex": filename}})
                    if doc:
                        status_data = {"status": "complete", "progress": 100, "message": "Processing complete! Video indexed."}
                        s3_client.put_object(Bucket=bucket, Key=status_key,
                            Body=json.dumps(status_data).encode(), ContentType="application/json")
                    else:
                        # Check S3 embeddings folder (Lambda uses underscores: input/file.mp4 -> input_file.mp4)
                        embeddings_prefix = f"embeddings/{s3_key.replace('/', '_')}/"
                        check = s3_client.list_objects_v2(Bucket=bucket, Prefix=embeddings_prefix, MaxKeys=1)
                        if check.get("KeyCount", 0) > 0:
                            status_data["progress"] = 70
                            status_data["message"] = "Embeddings generated. Ingesting into vector DB..."
                except Exception:
                    pass  # Keep existing status if check fails

        return status_data
    except Exception:
        return JSONResponse({"error": "Upload not found"}, status_code=404)


# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
