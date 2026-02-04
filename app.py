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
S3_BUCKET = os.environ.get("S3_BUCKET", "tl-brice-media")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
CLOUDFRONT_DOMAIN = os.environ.get("CLOUDFRONT_DOMAIN", "d2h48upmn4e6uy.cloudfront.net")

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
    query: str
    modalities: Optional[list] = None
    weights: Optional[dict] = None
    limit: int = 50
    video_id: Optional[str] = None
    fusion_method: str = "rrf"  # "rrf" or "weighted"


class SearchResult(BaseModel):
    """Single search result."""
    video_id: str
    segment_id: int
    start_time: float
    end_time: float
    s3_uri: str
    fusion_score: float
    modality_scores: dict
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None


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


@app.post("/api/search")
async def search(request: SearchRequest) -> list[SearchResult]:
    """
    Search for video segments.

    - **query**: Text search query
    - **modalities**: List of modalities ["visual", "audio", "transcription"]
    - **weights**: Custom weights per modality
    - **limit**: Max results (default 50)
    - **video_id**: Filter by specific video
    """
    client = get_search_client()

    results = client.search(
        query=request.query,
        modalities=request.modalities,
        weights=request.weights,
        limit=request.limit,
        video_id=request.video_id,
        fusion_method=request.fusion_method
    )

    # Add CloudFront URLs for fast video playback
    for result in results:
        s3_uri = result["s3_uri"]
        parsed = urlparse(s3_uri)
        key = parsed.path.lstrip("/")

        # Use proxy folder for faster video delivery (480p, ~90MB vs ~1.5GB)
        proxy_key = key.replace("WBD_project/Videos/", "WBD_project/Videos/proxy/")
        result["video_url"] = f"https://{CLOUDFRONT_DOMAIN}/{proxy_key}"

        # Thumbnail URL (we'll generate these separately)
        result["thumbnail_url"] = f"/api/thumbnail/{result['video_id']}/{result['segment_id']}"

    return results


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

    # Use proxy folder for faster thumbnail loading
    proxy_key = key.replace("WBD_project/Videos/", "WBD_project/Videos/proxy/")
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
