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
    """List videos in a specific index. backend=mongodb|s3vectors, index_mode=unified|multi."""
    client = get_search_client()

    if backend == "mongodb":
        db = client.db
        if index_mode == "unified":
            collection = db["unified-embeddings"]
        else:
            collection = db["visual_embeddings"]
    elif backend == "s3vectors":
        db = client.db
        collection = db["unified-embeddings"] if index_mode == "unified" else db["visual_embeddings"]
    else:
        return []

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

    # Add CloudFront URLs and human-readable names
    for video in videos:
        s3_uri = video.get("s3_uri", "")
        if s3_uri:
            # Normalize old S3 URIs to current bucket/path
            s3_uri = _normalize_s3_uri(s3_uri)
            video["s3_uri"] = s3_uri

            parsed = urlparse(s3_uri)
            key = parsed.path.lstrip("/")
            if key.startswith("input/"):
                key = key.replace("input/", "proxies/", 1)
            video["video_url"] = f"https://{CLOUDFRONT_DOMAIN}/{key}"

            # Extract readable name from proxy filename
            filename = os.path.basename(key)
            name_no_ext = os.path.splitext(filename)[0]
            video["name"] = name_no_ext.replace("_", " ").replace("-", " ")

    return videos


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
    """Find videos similar to a reference video."""
    body = await request.json()
    video_id = body.get("video_id")
    if not video_id:
        return JSONResponse({"error": "video_id required"}, status_code=400)

    search_client = get_search_client()
    mongodb = search_client.get_mongodb_client()

    ref_fp = mongodb.get_video_fingerprint(video_id)
    if not ref_fp:
        return JSONResponse({"error": f"No fingerprint found for video {video_id}"}, status_code=404)

    all_fps = mongodb.get_all_fingerprints()
    results = find_similar_videos(video_id, all_fps)

    # Helper to resolve video CloudFront URL from s3_uri
    def _resolve_video_url(vid_id):
        seg = mongodb.db[mongodb.COLLECTION_NAME].find_one(
            {"video_id": vid_id}, {"s3_uri": 1, "_id": 0}
        )
        if seg and CLOUDFRONT_DOMAIN:
            s3_uri = _normalize_s3_uri(seg.get("s3_uri", ""))
            if s3_uri:
                parsed = urlparse(s3_uri)
                key = parsed.path.lstrip("/")
                if key.startswith("input/"):
                    key = key.replace("input/", "proxies/", 1)
                return f"https://{CLOUDFRONT_DOMAIN}/{key}"
        return None

    # Add video URLs to results
    for r in results:
        url = _resolve_video_url(r["video_id"])
        if url:
            r["video_url"] = url
        # Add thumbnail URL if available
        fp = mongodb.get_video_fingerprint(r["video_id"])
        if fp and fp.get("thumbnail_key"):
            r["thumbnail_url"] = f"https://{CLOUDFRONT_DOMAIN}/{fp['thumbnail_key']}"

    # Resolve reference video URL
    ref_url = _resolve_video_url(video_id)
    ref_thumbnail = None
    if ref_fp.get("thumbnail_key"):
        ref_thumbnail = f"https://{CLOUDFRONT_DOMAIN}/{ref_fp['thumbnail_key']}"

    return {
        "reference": {
            "video_id": video_id,
            "name": ref_fp.get("video_name", video_id),
            "segment_count": ref_fp.get("segment_count", 0),
            "duration": ref_fp.get("total_duration", 0.0),
            "video_url": ref_url,
            "thumbnail_url": ref_thumbnail,
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
    return result


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
        },
        "compare": {
            "video_id": compare_id,
            "name": cmp_fp.get("video_name", compare_id) if cmp_fp else compare_id,
            "segment_count": cmp_fp.get("segment_count", 0) if cmp_fp else len(set(s["segment_id"] for s in cmp_segments)),
            "duration": cmp_fp.get("total_duration", 0.0) if cmp_fp else 0.0,
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

    output = io.StringIO()
    writer = csv.writer(output)
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


# =============================================
# Upload Endpoints
# =============================================

@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    settings: str = Form(default="{}")
):
    """Upload video file and trigger processing."""
    # Validate file type
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

    # Invoke Lambda asynchronously
    try:
        lambda_client = boto3.client("lambda", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        payload = {
            "s3_key": s3_key,
            "bucket": bucket,
            "segmentation_method": settings_dict.get("segmentation", "dynamic"),
            "min_duration_sec": settings_dict.get("duration", 4),
            "segment_length_sec": settings_dict.get("duration", 6),
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
