"""
Video Search API

FastAPI backend for multi-modal video search with Bedrock Marengo
and MongoDB Atlas vector search.
"""

import os
import sys
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from search_client import VideoSearchClient

# Configuration (set via environment variables)
MONGODB_URI = os.environ.get("MONGODB_URI")
S3_BUCKET = os.environ.get("S3_BUCKET", "your-media-bucket-name")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
CLOUDFRONT_DOMAIN = os.environ.get("CLOUDFRONT_DOMAIN", "xxxxx.cloudfront.net")

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
        # Add CloudFront URLs for fast video playback
        for result in results:
            s3_uri = result["s3_uri"]
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

        # Add CloudFront URLs for fast video playback
        for result in results:
            s3_uri = result["s3_uri"]
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


# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
