# 3D Cluster Compare View Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat video selection grid on the compare page with an interactive 3D cluster visualization where videos are grouped by embedding similarity, navigable with pan/zoom, and searchable semantically.

**Architecture:** New `src/clustering.py` module handles agglomerative clustering + 2D projection. The existing `/api/indexes/{backend}/{indexMode}/videos` endpoint is extended to return a `clusters` array alongside `videos`. Frontend replaces the `#compareVideoGrid` container with a CSS-transform-based 3D scene (pan/zoom/fly-to), search bar, and cluster selection panel. Existing NLE timeline, preview strip, and comparison logic are untouched.

**Tech Stack:** Python (scikit-learn for clustering, numpy for cosine/projection), FastAPI, vanilla JS/CSS (no WebGL), existing Bedrock Marengo embeddings (512d).

---

## Chunk 1: Backend — Clustering Module + API

### Task 1: Add scikit-learn dependency and create clustering module

**Files:**
- Modify: `requirements.txt`
- Create: `src/clustering.py`
- Create: `tests/test_clustering.py`

This task creates a standalone clustering module. It takes a list of video embedding vectors, groups them with agglomerative clustering (cosine distance, threshold 0.3), computes cluster centroids, average within-cluster similarity, and 2D positions via t-SNE.

**Context:** Each MongoDB document stores a 512d `embedding` vector per segment per modality. For clustering, we need one vector per video. The approach: for each video, average all its segment embeddings within a single modality (visual, as the representative modality), producing one 512d vector per video. Then cluster those vectors.

**Deviation from spec:** The spec says "combined (fused) embedding vectors." We use visual-only because: (1) computing a fused vector requires weighting visual+audio+transcription, but there's no canonical weighting for clustering (only for search with dynamic weights); (2) visual embeddings are the most consistently available across videos; (3) the clustering quality from visual-only is sufficient for grouping similar videos. This is a deliberate trade-off for simplicity.

- [ ] **Step 1: Add scikit-learn to requirements.txt**

Append to the end of `requirements.txt`:

```
# Clustering for compare page
scikit-learn>=1.3.0
```

Note: We do NOT need `umap-learn`. scikit-learn's `TSNE` is sufficient for projecting a handful of cluster centroids to 2D (we're clustering videos, not thousands of high-dim points). Keeping dependencies minimal.

- [ ] **Step 2: Write the test file**

Create `tests/test_clustering.py`:

```python
"""Tests for video clustering module."""
import numpy as np
import pytest
from src.clustering import cluster_videos, compute_2d_positions


def _make_video_embedding(base, noise_scale=0.05):
    """Create a 512d embedding near a base vector with small noise."""
    rng = np.random.RandomState(42)
    return (np.array(base) + rng.randn(512) * noise_scale).tolist()


# Two clusters: 3 similar videos + 1 outlier
# Use explicitly orthogonal vectors to guarantee separation
BASE_A = np.zeros(512); BASE_A[0] = 1.0; BASE_A = BASE_A.tolist()
BASE_B = np.zeros(512); BASE_B[256] = 1.0; BASE_B = BASE_B.tolist()  # orthogonal to A

VIDEOS = {
    "vid_a1": _make_video_embedding(BASE_A, 0.02),
    "vid_a2": _make_video_embedding(BASE_A, 0.02),
    "vid_a3": _make_video_embedding(BASE_A, 0.02),
    "vid_b1": _make_video_embedding(BASE_B, 0.02),
}


class TestClusterVideos:
    def test_returns_list_of_cluster_dicts(self):
        clusters = cluster_videos(VIDEOS)
        assert isinstance(clusters, list)
        assert len(clusters) >= 1
        for c in clusters:
            assert "id" in c
            assert "video_ids" in c
            assert "centroid" in c
            assert "avg_similarity" in c
            assert isinstance(c["video_ids"], list)
            assert len(c["centroid"]) == 512

    def test_groups_similar_videos_together(self):
        clusters = cluster_videos(VIDEOS)
        # Find the cluster containing vid_a1
        a_cluster = next(c for c in clusters if "vid_a1" in c["video_ids"])
        # All three A videos should be in the same cluster
        assert "vid_a2" in a_cluster["video_ids"]
        assert "vid_a3" in a_cluster["video_ids"]
        # B should NOT be in the A cluster
        assert "vid_b1" not in a_cluster["video_ids"]

    def test_singleton_cluster_for_outlier(self):
        clusters = cluster_videos(VIDEOS)
        b_cluster = next(c for c in clusters if "vid_b1" in c["video_ids"])
        assert len(b_cluster["video_ids"]) == 1

    def test_avg_similarity_range(self):
        clusters = cluster_videos(VIDEOS)
        for c in clusters:
            if len(c["video_ids"]) > 1:
                assert 0.0 <= c["avg_similarity"] <= 1.0
            else:
                # Singletons have no pairwise similarity
                assert c["avg_similarity"] == 1.0

    def test_empty_input(self):
        clusters = cluster_videos({})
        assert clusters == []

    def test_single_video(self):
        clusters = cluster_videos({"only_vid": BASE_A})
        assert len(clusters) == 1
        assert clusters[0]["video_ids"] == ["only_vid"]

    def test_custom_distance_threshold(self):
        # Very tight threshold — should split into more clusters
        clusters_tight = cluster_videos(VIDEOS, distance_threshold=0.01)
        clusters_loose = cluster_videos(VIDEOS, distance_threshold=0.9)
        assert len(clusters_tight) >= len(clusters_loose)


class TestCompute2dPositions:
    def test_returns_positions_for_each_cluster(self):
        clusters = cluster_videos(VIDEOS)
        positions = compute_2d_positions(clusters)
        assert len(positions) == len(clusters)
        for cid, pos in positions.items():
            assert "x" in pos and "y" in pos
            assert 0.0 <= pos["x"] <= 1.0
            assert 0.0 <= pos["y"] <= 1.0

    def test_single_cluster_centered(self):
        clusters = cluster_videos({"vid": BASE_A})
        positions = compute_2d_positions(clusters)
        # Single cluster should be at center (0.5, 0.5)
        pos = list(positions.values())[0]
        assert pos["x"] == 0.5
        assert pos["y"] == 0.5

    def test_two_clusters_separated(self):
        clusters = cluster_videos(VIDEOS)
        if len(clusters) >= 2:
            positions = compute_2d_positions(clusters)
            coords = list(positions.values())
            # Two clusters shouldn't be at exact same position
            dist = ((coords[0]["x"] - coords[1]["x"])**2 + (coords[0]["y"] - coords[1]["y"])**2)**0.5
            assert dist > 0.05  # Meaningfully separated
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /Users/bpenven/Documents/Code/multi-modal-video-search
python -m pytest tests/test_clustering.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.clustering'`

- [ ] **Step 4: Implement the clustering module**

Create `src/clustering.py`:

```python
"""
Video clustering module.

Groups videos by embedding similarity using agglomerative clustering
with cosine distance. Produces cluster objects with centroids,
avg similarity, and 2D positions for visualization.
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normalized = vectors / norms
    return normalized @ normalized.T


def cluster_videos(
    video_embeddings: dict[str, list],
    distance_threshold: float = 0.3
) -> list[dict]:
    """
    Cluster videos by embedding similarity.

    Args:
        video_embeddings: {video_id: 512d_embedding_list}
        distance_threshold: Cosine distance threshold for grouping (0.3 = 0.7 similarity)

    Returns:
        List of cluster dicts with keys:
        - id: "cluster_0", "cluster_1", ...
        - video_ids: list of video_id strings
        - centroid: 512d list (mean of member embeddings)
        - avg_similarity: mean pairwise cosine similarity within cluster (1.0 for singletons)
    """
    if not video_embeddings:
        return []

    video_ids = list(video_embeddings.keys())
    vectors = np.array([video_embeddings[vid] for vid in video_ids], dtype=np.float64)

    if len(video_ids) == 1:
        return [{
            "id": "cluster_0",
            "video_ids": video_ids,
            "centroid": vectors[0].tolist(),
            "avg_similarity": 1.0
        }]

    # Agglomerative clustering with cosine distance
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(vectors)

    # Build clusters
    sim_matrix = cosine_similarity_matrix(vectors)
    clusters = []

    for label in sorted(set(labels)):
        member_mask = labels == label
        member_indices = np.where(member_mask)[0]
        member_ids = [video_ids[i] for i in member_indices]
        member_vectors = vectors[member_indices]

        centroid = member_vectors.mean(axis=0)

        # Average pairwise similarity within cluster
        if len(member_indices) > 1:
            sub_sim = sim_matrix[np.ix_(member_indices, member_indices)]
            # Upper triangle (excluding diagonal)
            triu = np.triu_indices(len(member_indices), k=1)
            avg_sim = float(sub_sim[triu].mean())
        else:
            avg_sim = 1.0

        clusters.append({
            "id": f"cluster_{label}",
            "video_ids": member_ids,
            "centroid": centroid.tolist(),
            "avg_similarity": round(avg_sim, 4)
        })

    return clusters


def compute_2d_positions(clusters: list[dict]) -> dict[str, dict]:
    """
    Project cluster centroids to 2D for visualization.

    Args:
        clusters: Output from cluster_videos()

    Returns:
        {cluster_id: {"x": 0.0-1.0, "y": 0.0-1.0}}
    """
    if not clusters:
        return {}

    if len(clusters) == 1:
        return {clusters[0]["id"]: {"x": 0.5, "y": 0.5}}

    centroids = np.array([c["centroid"] for c in clusters], dtype=np.float64)

    if len(clusters) == 2:
        # t-SNE needs perplexity < n_samples; for 2 points just spread them
        return {
            clusters[0]["id"]: {"x": 0.3, "y": 0.5},
            clusters[1]["id"]: {"x": 0.7, "y": 0.5},
        }

    # t-SNE projection
    perplexity = min(5, len(clusters) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        metric="cosine",
        init="random"
    )
    coords_2d = tsne.fit_transform(centroids)

    # Normalize to 0-1 with padding
    padding = 0.1
    for dim in range(2):
        col = coords_2d[:, dim]
        mn, mx = col.min(), col.max()
        span = mx - mn if mx != mn else 1.0
        coords_2d[:, dim] = padding + (col - mn) / span * (1 - 2 * padding)

    positions = {}
    for i, c in enumerate(clusters):
        positions[c["id"]] = {
            "x": round(float(coords_2d[i, 0]), 4),
            "y": round(float(coords_2d[i, 1]), 4)
        }

    return positions
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/bpenven/Documents/Code/multi-modal-video-search
pip install scikit-learn>=1.3.0
python -m pytest tests/test_clustering.py -v
```

Expected: All 10 tests PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/clustering.py tests/test_clustering.py
git commit -m "feat: add video clustering module with agglomerative cosine clustering"
```

---

### Task 2: Extend videos API endpoint to return clusters

**Files:**
- Modify: `app.py:398-458`
- Modify: `src/clustering.py` (add `auto_name_cluster` helper)

This task modifies the existing `/api/indexes/{backend}/{indexMode}/videos` endpoint to:
1. Fetch one representative embedding per video (average of visual segment embeddings)
2. Call `cluster_videos()` + `compute_2d_positions()`
3. Return `{ videos: [...], clusters: [...] }` instead of a bare array

**Context:** Currently the endpoint returns a plain array of video objects. We need to wrap it in an object and add `clusters`. The frontend already handles both `data.videos` and `data` (array) — see `static/index.html:8399`: `const videos = data.videos || data || [];`

- [ ] **Step 1: Add auto-naming helper to clustering.py**

Append to `src/clustering.py`:

```python
def auto_name_cluster(video_names: list[str]) -> str:
    """
    Generate a human-readable cluster name from video names.

    Finds the longest common prefix among the names. Falls back to
    "Cluster N" style naming if no meaningful prefix is found.
    """
    if not video_names:
        return "Empty Cluster"
    if len(video_names) == 1:
        return video_names[0]

    # Find longest common prefix of all names
    prefix = video_names[0]
    for name in video_names[1:]:
        while not name.lower().startswith(prefix.lower()) and len(prefix) > 0:
            prefix = prefix[:-1]

    # Clean up trailing separators
    prefix = prefix.rstrip(" _-")

    if len(prefix) >= 3:
        return f"{prefix} Versions"

    # Fallback per spec: numbered cluster name
    return None  # caller assigns "Cluster N" with index
```

- [ ] **Step 2: Modify the videos endpoint in app.py**

Replace `app.py:398-458` — the `list_index_videos` function. The new version:
1. Fetches videos as before
2. Fetches embeddings for each video (average of visual segments)
3. Clusters them
4. Returns `{ videos: [...], clusters: [...] }`

```python
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
    if len(video_embeddings) >= 2:
        from clustering import cluster_videos, compute_2d_positions, auto_name_cluster

        clusters = cluster_videos(video_embeddings)
        positions = compute_2d_positions(clusters)

        # Build video name lookup
        name_lookup = {v.get("video_id", ""): v.get("name", "") for v in videos}

        for i, c in enumerate(clusters):
            c["position"] = positions.get(c["id"], {"x": 0.5, "y": 0.5})
            auto = auto_name_cluster([name_lookup.get(vid, vid) for vid in c["video_ids"]])
            c["name"] = auto if auto else f"Cluster {i + 1}"

        clusters_response = clusters
    elif len(video_embeddings) == 1:
        from clustering import cluster_videos, auto_name_cluster
        vid_id = list(video_embeddings.keys())[0]
        name_lookup = {v.get("video_id", ""): v.get("name", "") for v in videos}
        clusters_response = [{
            "id": "cluster_0",
            "video_ids": [vid_id],
            "centroid": video_embeddings[vid_id],
            "avg_similarity": 1.0,
            "position": {"x": 0.5, "y": 0.5},
            "name": name_lookup.get(vid_id, vid_id)
        }]

    return {"videos": videos, "clusters": clusters_response}
```

- [ ] **Step 3: Verify numpy import is present in app.py**

Check `app.py` imports. The clustering code uses `import numpy as np` inside the function body (lazy import). No top-level change needed. The `from clustering import ...` is also done lazily inside the function to avoid import errors when the module doesn't exist. Note: `app.py` uses `sys.path.insert(0, ...)` for the `src/` directory, so imports use `from clustering import ...` (not `from src.clustering`).

- [ ] **Step 4: Test manually with curl**

```bash
# Start the app locally
cd /Users/bpenven/Documents/Code/multi-modal-video-search
uvicorn app:app --reload --port 8000 &

# Test the endpoint
curl -s http://localhost:8000/api/indexes/mongodb/multi/videos | python3 -m json.tool | head -60
```

Expected: JSON object with `"videos": [...]` and `"clusters": [...]` keys. Clusters should have `id`, `video_ids`, `centroid` (truncated), `avg_similarity`, `position`, `name`.

- [ ] **Step 5: Commit**

```bash
git add app.py src/clustering.py
git commit -m "feat: extend videos API to return embedding-based clusters with 2D positions"
```

---

### Task 3: Add semantic search endpoint

**Files:**
- Modify: `app.py` (add new endpoint after the videos endpoint)

The frontend search bar needs a backend endpoint to compute similarity between a text query and cluster centroids. This is a thin endpoint that embeds the query text and returns cosine similarities.

- [ ] **Step 1: Add the search endpoint**

Add after the `list_index_videos` function in `app.py`:

```python
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
```

- [ ] **Step 2: Test with curl**

```bash
# Requires the app to be running
curl -s -X POST http://localhost:8000/api/clusters/search \
  -H "Content-Type: application/json" \
  -d '{"query": "nature documentary", "centroids": {"c0": [0.1, 0.2]}}' \
  | python3 -m json.tool
```

Expected: `{"scores": {"c0": <float>}}`

Note: The centroid in the test is only 2d so the score won't be meaningful, but it verifies the endpoint works.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add cluster semantic search endpoint"
```

---

## Chunk 2: Frontend — 3D Cluster Scene + Navigation

### Task 4: Replace video grid with 3D cluster view (CSS + HTML)

**Files:**
- Modify: `static/index.html` — CSS section (~lines 3246-3345) and HTML section (~lines 5152-5182)

This task replaces the flat `.compare-video-grid` with the 3D cluster viewport HTML structure and all required CSS. No JS yet — just the static structure.

**Context:** The compare page HTML lives at `static/index.html:5152-5266`. The `.compare-ref-bar` div contains the backend/index mode toggles, the `#compareVideoGrid` container, and the Compare button. We're replacing the grid + button with the cluster viewport + selection panel, but keeping the backend/index mode toggles and the timeline section (`#compareDiffContainer`) below.

- [ ] **Step 1: Add cluster view CSS**

After the existing `.compare-video-grid-empty` rule (~line 3344), add these new CSS rules. **Do not remove the existing compare-video-card CSS** — it's still used for the selection panel's video cards inside clusters.

```css
/* ── 3D Cluster View ── */
.cluster-viewport {
    position: relative;
    height: 500px;
    overflow: hidden;
    cursor: grab;
    border-radius: 10px;
    border: 1px solid var(--border-light);
    background: var(--bg-body);
    margin-bottom: 12px;
    perspective: 600px;
}
.cluster-viewport.dragging { cursor: grabbing; }

.cluster-scene {
    position: absolute;
    width: 100%; height: 100%;
    transform-origin: 50% 50%;
    will-change: transform;
}

.cluster-grid-floor {
    position: absolute;
    width: 1200px; height: 1200px;
    left: 50%; top: 55%;
    transform: translate(-50%, -50%) rotateX(70deg) rotateZ(15deg);
    background:
        linear-gradient(var(--border-light) 1px, transparent 1px),
        linear-gradient(90deg, var(--border-light) 1px, transparent 1px);
    background-size: 60px 60px;
    border: 1px solid rgba(69,66,63,0.2);
    opacity: 0.3;
}

.cluster-bubble {
    position: absolute;
    border-radius: 50%;
    border: 1.5px solid rgba(255,255,255,0.08);
    background: radial-gradient(ellipse at 30% 30%, rgba(255,255,255,0.04), transparent 70%);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: border-color 0.3s, background 0.3s, box-shadow 0.3s;
}
.cluster-bubble:hover {
    border-color: rgba(0,220,130,0.4);
    background: radial-gradient(ellipse at 30% 30%, rgba(0,220,130,0.06), transparent 70%);
}
.cluster-bubble.selected {
    border-color: rgba(0,220,130,0.6);
    background: radial-gradient(ellipse at 30% 30%, rgba(0,220,130,0.1), transparent 70%);
    box-shadow: 0 0 40px rgba(0,220,130,0.15);
}
.cluster-bubble.search-hit {
    border-color: rgba(0,220,130,0.8) !important;
    box-shadow: 0 0 30px rgba(0,220,130,0.25), 0 0 60px rgba(0,220,130,0.1) !important;
}
.cluster-bubble.search-partial {
    border-color: rgba(250,186,23,0.6) !important;
    box-shadow: 0 0 25px rgba(250,186,23,0.2) !important;
}
.cluster-bubble.search-dimmed {
    opacity: 0.2 !important;
    filter: grayscale(0.5);
}

.cluster-bubble-label {
    position: absolute;
    bottom: -22px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 9px;
    color: var(--text-tertiary);
    white-space: nowrap;
    text-align: center;
    pointer-events: none;
}
.cluster-bubble.selected .cluster-bubble-label { color: var(--accent); }

.cluster-bubble-count {
    font-size: 10px;
    color: rgba(255,255,255,0.5);
    font-family: 'IBM Plex Mono', monospace;
    pointer-events: none;
}

.cluster-video-dot {
    position: absolute;
    width: 8px; height: 8px;
    border-radius: 50%;
    transition: transform 0.3s;
}
.cluster-bubble:hover .cluster-video-dot { transform: scale(1.3); }

/* Cluster view fixed overlays */
.cluster-search-bar {
    position: absolute;
    top: 56px; left: 16px;
    z-index: 20;
    display: flex;
    width: 300px;
}
.cluster-search-input {
    flex: 1;
    padding: 7px 10px 7px 28px;
    border-radius: 6px 0 0 6px;
    background: rgba(35,34,33,0.9);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    border-right: none;
    color: var(--text-primary);
    font-size: 11px;
    font-family: 'Noto Sans', system-ui, sans-serif;
    outline: none;
}
.cluster-search-input:focus { border-color: var(--accent); }
.cluster-search-input::placeholder { color: var(--text-tertiary); }
.cluster-search-icon {
    position: absolute;
    left: 9px; top: 50%;
    transform: translateY(-50%);
    font-size: 11px;
    color: var(--text-tertiary);
    pointer-events: none;
}
.cluster-search-go {
    padding: 7px 12px;
    border-radius: 0 6px 6px 0;
    background: var(--accent-muted);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-size: 10px;
    font-weight: 600;
    cursor: pointer;
}

.cluster-search-results {
    position: absolute;
    top: 88px; left: 16px;
    z-index: 20;
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}
.cluster-search-chip {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 9px;
    font-family: 'IBM Plex Mono', monospace;
    cursor: pointer;
    border: 1px solid;
}
.cluster-search-chip.match { background: var(--accent-muted); border-color: var(--accent); color: var(--accent); }
.cluster-search-chip.partial { background: rgba(250,186,23,0.10); border-color: var(--warning); color: var(--warning); }
.cluster-search-clear {
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 9px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    color: var(--text-tertiary);
    cursor: pointer;
}

.cluster-zoom-controls {
    position: absolute;
    bottom: 12px; right: 12px;
    z-index: 20;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 8px;
    border-radius: 8px;
    background: rgba(35,34,33,0.85);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
}
.cluster-zoom-btn {
    width: 24px; height: 24px;
    border-radius: 5px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    color: var(--text-tertiary);
    font-size: 13px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
}
.cluster-zoom-btn:hover { border-color: var(--accent); color: var(--accent); }
.cluster-zoom-indicator {
    font-size: 10px;
    color: var(--text-tertiary);
    font-family: 'IBM Plex Mono', monospace;
    min-width: 30px;
    text-align: center;
}
.cluster-zoom-reset {
    font-size: 9px;
    color: var(--text-tertiary);
    cursor: pointer;
    padding: 2px 5px;
    border-radius: 3px;
    border: 1px solid transparent;
    background: none;
    font-family: 'IBM Plex Mono', monospace;
}
.cluster-zoom-reset:hover { border-color: var(--border); color: var(--text-primary); }

.cluster-scene.animating {
    transition: transform 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

/* Selection panel — right side of cluster viewport */
.cluster-panel {
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 280px;
    background: var(--bg-surface);
    border-left: 1px solid var(--border-light);
    display: flex;
    flex-direction: column;
    z-index: 25;
    transition: transform 0.3s;
}
.cluster-panel.hidden { transform: translateX(100%); }
.cluster-panel-header {
    padding: 12px 14px;
    border-bottom: 1px solid var(--border-light);
}
.cluster-panel-title { font-size: 12px; font-weight: 600; }
.cluster-panel-subtitle { font-size: 9px; color: var(--text-tertiary); margin-top: 2px; }
.cluster-panel-body { flex: 1; overflow-y: auto; padding: 10px; }
.cluster-panel-footer {
    padding: 10px 14px;
    border-top: 1px solid var(--border-light);
    display: flex;
    align-items: center;
    gap: 8px;
}
.cluster-panel-video {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px;
    border-radius: 6px;
    border: 1px solid var(--border);
    margin-bottom: 5px;
    cursor: pointer;
    transition: all 0.15s;
}
.cluster-panel-video:hover { border-color: var(--border-hover); }
.cluster-panel-video.checked { border-color: var(--accent); background: rgba(0,220,130,0.05); }
.cluster-panel-checkbox {
    width: 14px; height: 14px;
    border-radius: 3px;
    border: 1.5px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    color: transparent;
    flex-shrink: 0;
}
.cluster-panel-video.checked .cluster-panel-checkbox {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(0,220,130,0.15);
}
.cluster-panel-thumb {
    width: 44px; height: 28px;
    border-radius: 3px;
    flex-shrink: 0;
    object-fit: cover;
    background: var(--bg-elevated);
}
.cluster-panel-vname { font-size: 10px; color: var(--text-primary); }
.cluster-panel-vmeta { font-size: 8px; color: var(--text-tertiary); font-family: 'IBM Plex Mono', monospace; margin-top: 1px; }
.cluster-panel-vscore { font-size: 10px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; flex-shrink: 0; }
.cluster-cross-select {
    font-size: 9px;
    color: var(--text-tertiary);
    padding: 4px 8px;
    border-radius: 4px;
    background: rgba(255,255,255,0.03);
    margin-bottom: 8px;
    display: none;
}
```

- [ ] **Step 2: Replace compare grid HTML with cluster viewport**

Replace the HTML in `static/index.html` from the `<label>Select Reference Video</label>` line through the Compare button (lines ~5179-5181) with:

```html
<div class="cluster-viewport" id="clusterViewport">
    <!-- Fixed overlays -->
    <div style="position:absolute;top:12px;left:14px;z-index:20;">
        <div style="font-size:14px;font-weight:600;color:var(--text-primary);">Video Clusters</div>
        <div style="font-size:10px;color:var(--text-tertiary);">Drag to pan · Scroll to zoom · Click a cluster</div>
    </div>

    <div class="cluster-search-bar">
        <span class="cluster-search-icon">&#128269;</span>
        <input class="cluster-search-input" id="clusterSearchInput" type="text"
               placeholder='Search videos...' spellcheck="false">
        <button class="cluster-search-go" id="clusterSearchGo">Find</button>
    </div>
    <div class="cluster-search-results" id="clusterSearchResults"></div>

    <div class="cluster-zoom-controls">
        <button class="cluster-zoom-btn" id="clusterZoomOut">&minus;</button>
        <span class="cluster-zoom-indicator" id="clusterZoomIndicator">1.0x</span>
        <button class="cluster-zoom-btn" id="clusterZoomIn">+</button>
        <button class="cluster-zoom-reset" id="clusterZoomReset">reset</button>
    </div>

    <!-- Transformable scene -->
    <div class="cluster-scene" id="clusterScene">
        <div class="cluster-grid-floor"></div>
        <svg id="clusterLines" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:1;"></svg>
        <div id="clusterBubbles"></div>
    </div>

    <!-- Selection panel -->
    <div class="cluster-panel hidden" id="clusterPanel">
        <div class="cluster-panel-header">
            <div class="cluster-panel-title" id="clusterPanelTitle"></div>
            <div class="cluster-panel-subtitle" id="clusterPanelSubtitle"></div>
        </div>
        <div class="cluster-panel-body">
            <div class="cluster-cross-select" id="clusterCrossSelect"></div>
            <div id="clusterPanelVideos"></div>
        </div>
        <div class="cluster-panel-footer">
            <button id="clusterCompareBtn" class="compare-action-btn" onclick="compareCheckedVideos()" disabled>Compare</button>
            <span style="font-size:9px;color:var(--text-tertiary)">Opens NLE timeline</span>
        </div>
    </div>
</div>
```

Also remove the old `#compareBtn` button that was after the grid (line ~5181), since the Compare button now lives inside the cluster panel footer. Keep the old `#compareVideoGrid` div but set it to `display:none` — it's referenced by `updateRefBadge()` which we'll update in the next task.

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat: add 3D cluster viewport CSS and HTML structure"
```

---

### Task 5: Implement cluster rendering, navigation, and selection (JS)

**Files:**
- Modify: `static/index.html` — JS section (replace `loadCompareVideos` and add cluster JS)

This is the main frontend task. It replaces `loadCompareVideos()` with cluster-based rendering, implements pan/zoom/fly-to navigation, cluster selection with panel population, and wires up the search bar.

**Context:**
- `compareCheckedIds` (line 8336) — ordered list of checked video IDs. This is the interface between cluster selection and the existing `compareCheckedVideos()` → `showTimelineDiff()` pipeline. We keep writing to this same variable.
- `updateCompareBtn()` (line 8338) — updates the Compare button text/disabled state. We need to update this to target the new `#clusterCompareBtn` instead of `#compareBtn`.
- `updateRefBadge()` (line 8345) — updates REF badge on video cards. We need to update this to work within the cluster panel.
- The existing `compareCheckedVideos()` (line 8470) calls `showTimelineDiff()` and is already correct — no changes needed.

- [ ] **Step 1: Update updateCompareBtn to target new button**

Replace `updateCompareBtn()` (lines 8338-8343):

```javascript
function updateCompareBtn() {
    const btn = document.getElementById('clusterCompareBtn') || document.getElementById('compareBtn');
    if (!btn) return;
    btn.disabled = compareCheckedIds.length < 2;
    btn.textContent = compareCheckedIds.length > 0 ? `Compare (${compareCheckedIds.length})` : 'Compare';
}
```

- [ ] **Step 2: Update updateRefBadge to work within cluster panel**

Replace `updateRefBadge()` (lines 8345-8355):

```javascript
function updateRefBadge() {
    const panel = document.getElementById('clusterPanelVideos');
    if (!panel) return;
    panel.querySelectorAll('.cluster-panel-video').forEach(card => {
        const badge = card.querySelector('.panel-ref-badge');
        if (badge) badge.style.display = 'none';
    });
    if (compareCheckedIds.length > 0) {
        const refCard = panel.querySelector(`.cluster-panel-video[data-video-id="${compareCheckedIds[0]}"]`);
        if (refCard) {
            const badge = refCard.querySelector('.panel-ref-badge');
            if (badge) badge.style.display = 'inline';
        }
    }
}
```

- [ ] **Step 3: Replace loadCompareVideos with cluster rendering**

Replace the entire `loadCompareVideos()` function (lines 8385-8468) with the cluster-based version. This is a large function — it handles:

1. Fetch videos + clusters from API
2. Render cluster bubbles with video dots + labels
3. Render similarity lines between clusters
4. Set up pan/zoom/fly-to navigation
5. Handle cluster click → populate panel
6. Handle video checkbox in panel → update `compareCheckedIds`

```javascript
// ── Cluster state ──
let clusterData = [];      // cluster objects from API
let clusterVideos = [];    // all video objects from API
let clusterPanX = 0, clusterPanY = 0, clusterZoom = 1.0;
const CLUSTER_ZOOM_MIN = 0.3, CLUSTER_ZOOM_MAX = 4.0;
let clusterDragging = false, clusterDragStartX, clusterDragStartY;
let selectedClusterId = null;

// Track colors for video dots
const DOT_COLORS = ['#00DC82','#6CD5FD','#FABA17','#FF6B6B','#A78BFA','#34D399','#F472B6'];

function applyClusterTransform() {
    const scene = document.getElementById('clusterScene');
    if (!scene) return;
    scene.style.transform = `translate(${clusterPanX}px, ${clusterPanY}px) scale(${clusterZoom})`;
    scene.style.transformOrigin = '50% 50%';
    const indicator = document.getElementById('clusterZoomIndicator');
    if (indicator) indicator.textContent = clusterZoom.toFixed(1) + 'x';
}

function flyToCluster(cluster) {
    const vp = document.getElementById('clusterViewport');
    const scene = document.getElementById('clusterScene');
    if (!vp || !scene) return;
    const rect = vp.getBoundingClientRect();
    const cx = cluster.position.x * rect.width;
    const cy = cluster.position.y * rect.height;

    scene.classList.add('animating');
    clusterZoom = 1.8;
    clusterPanX = rect.width / 2 - cx * clusterZoom;
    clusterPanY = rect.height / 2 - cy * clusterZoom;
    applyClusterTransform();
    setTimeout(() => scene.classList.remove('animating'), 650);
}

function flyToFitClusters(matched) {
    if (matched.length === 0) return;
    if (matched.length === 1) { flyToCluster(matched[0]); return; }
    const vp = document.getElementById('clusterViewport');
    const scene = document.getElementById('clusterScene');
    if (!vp || !scene) return;
    const rect = vp.getBoundingClientRect();

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const c of matched) {
        minX = Math.min(minX, c.position.x);
        maxX = Math.max(maxX, c.position.x);
        minY = Math.min(minY, c.position.y);
        maxY = Math.max(maxY, c.position.y);
    }
    const cx = (minX + maxX) / 2 * rect.width;
    const cy = (minY + maxY) / 2 * rect.height;
    const spanX = (maxX - minX) + 0.2;
    const spanY = (maxY - minY) + 0.2;
    const fitZoom = Math.min(1.6, 1 / Math.max(spanX, spanY));

    scene.classList.add('animating');
    clusterZoom = Math.max(CLUSTER_ZOOM_MIN, Math.min(CLUSTER_ZOOM_MAX, fitZoom));
    clusterPanX = rect.width / 2 - cx * clusterZoom;
    clusterPanY = rect.height / 2 - cy * clusterZoom;
    applyClusterTransform();
    setTimeout(() => scene.classList.remove('animating'), 650);
}

function selectCluster(clusterId) {
    selectedClusterId = clusterId;
    // Highlight bubble
    document.querySelectorAll('.cluster-bubble').forEach(b => b.classList.remove('selected'));
    const bubble = document.querySelector(`.cluster-bubble[data-cluster-id="${clusterId}"]`);
    if (bubble) bubble.classList.add('selected');

    const cluster = clusterData.find(c => c.id === clusterId);
    if (!cluster) return;

    // Populate panel
    const panel = document.getElementById('clusterPanel');
    const title = document.getElementById('clusterPanelTitle');
    const subtitle = document.getElementById('clusterPanelSubtitle');
    const body = document.getElementById('clusterPanelVideos');
    const crossSelect = document.getElementById('clusterCrossSelect');

    title.textContent = cluster.name;
    subtitle.textContent = `${cluster.video_ids.length} video${cluster.video_ids.length > 1 ? 's' : ''} · avg ${Math.round(cluster.avg_similarity * 100)}% similarity`;
    panel.classList.remove('hidden');

    // Cross-cluster selection indicator
    const otherCount = compareCheckedIds.filter(id => !cluster.video_ids.includes(id)).length;
    if (otherCount > 0) {
        crossSelect.style.display = 'block';
        crossSelect.textContent = `${otherCount} selected from other clusters`;
    } else {
        crossSelect.style.display = 'none';
    }

    body.innerHTML = '';
    const videoLookup = {};
    clusterVideos.forEach(v => { videoLookup[v.video_id] = v; });

    cluster.video_ids.forEach(vid => {
        const v = videoLookup[vid] || { video_id: vid, name: vid };
        const isChecked = compareCheckedIds.includes(vid);
        const unplayable = isUnplayableFile(v);
        const techVideo = v.technical_metadata?.video;
        const codecTag = techVideo ? [techVideo.codec, techVideo.resolution].filter(Boolean).join(' · ') : '';

        const card = document.createElement('div');
        card.className = 'cluster-panel-video' + (isChecked ? ' checked' : '');
        card.dataset.videoId = vid;

        card.innerHTML = `
            <span class="cluster-panel-checkbox">&#10003;</span>
            <img class="cluster-panel-thumb" src="/api/thumbnail/${encodeURIComponent(vid)}/1"
                 onerror="this.style.background='var(--bg-elevated)';this.removeAttribute('src')">
            <div style="flex:1;min-width:0">
                <div class="cluster-panel-vname">${escapeHtml(v.name || vid)}
                    <span class="panel-ref-badge" style="display:${compareCheckedIds[0] === vid ? 'inline' : 'none'};font-size:7px;font-weight:700;background:var(--accent);color:var(--bg-body);padding:1px 4px;border-radius:2px;margin-left:3px">REF</span>
                </div>
                <div class="cluster-panel-vmeta">${codecTag}${unplayable ? ' · <span style="color:var(--error)">no proxy</span>' : ''}</div>
            </div>
        `;

        card.addEventListener('click', () => {
            const idx = compareCheckedIds.indexOf(vid);
            if (idx >= 0) {
                compareCheckedIds.splice(idx, 1);
                card.classList.remove('checked');
            } else {
                if (compareCheckedIds.length >= 9) {
                    alert('Maximum 8 comparison videos (9 total including reference).');
                    return;
                }
                compareCheckedIds.push(vid);
                card.classList.add('checked');
            }
            compareRefVideoId = compareCheckedIds[0] || null;
            updateRefBadge();
            updateCompareBtn();
            // Update cross-cluster count
            const otherNow = compareCheckedIds.filter(id => !cluster.video_ids.includes(id)).length;
            crossSelect.style.display = otherNow > 0 ? 'block' : 'none';
            crossSelect.textContent = otherNow > 0 ? `${otherNow} selected from other clusters` : '';
        });

        body.appendChild(card);
    });
}

function loadCompareVideos() {
    const viewport = document.getElementById('clusterViewport');
    const bubbles = document.getElementById('clusterBubbles');
    const lines = document.getElementById('clusterLines');
    if (!viewport || !bubbles) return;

    // Reset state
    compareCheckedIds = [];
    compareRefVideoId = null;
    selectedClusterId = null;
    clusterPanX = 0; clusterPanY = 0; clusterZoom = 1.0;
    updateCompareBtn();
    document.getElementById('clusterPanel')?.classList.add('hidden');

    bubbles.innerHTML = '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--text-tertiary);font-size:12px">Loading clusters...</div>';
    if (lines) lines.innerHTML = '';

    fetch(`/api/indexes/${compareBackend}/${compareIndexMode}/videos`)
        .then(r => r.json())
        .then(data => {
            clusterVideos = data.videos || data || [];
            clusterData = data.clusters || [];

            // Store video URLs
            clusterVideos.forEach(v => {
                if (v.video_url) compareVideoUrls[v.video_id] = v.video_url;
            });

            bubbles.innerHTML = '';

            if (clusterData.length === 0) {
                bubbles.innerHTML = '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--text-tertiary);font-size:12px">No clusters found. Upload and process videos first.</div>';
                return;
            }

            // Render cluster bubbles
            const vpRect = viewport.getBoundingClientRect();
            clusterData.forEach((cluster, ci) => {
                const count = cluster.video_ids.length;
                const size = Math.min(220, 60 + count * 30);
                const x = cluster.position.x * 100;
                const y = cluster.position.y * 100;

                const bubble = document.createElement('div');
                bubble.className = 'cluster-bubble';
                bubble.dataset.clusterId = cluster.id;
                bubble.style.cssText = `width:${size}px;height:${size}px;left:${x}%;top:${y}%;z-index:${Math.max(1, 10 - ci)};margin-left:-${size/2}px;margin-top:-${size/2}px;`;

                // Video dots
                cluster.video_ids.forEach((vid, vi) => {
                    const dot = document.createElement('div');
                    dot.className = 'cluster-video-dot';
                    const angle = (vi / cluster.video_ids.length) * Math.PI * 2;
                    const r = 25;
                    dot.style.cssText = `background:${DOT_COLORS[vi % DOT_COLORS.length]};top:${50 + Math.sin(angle) * r}%;left:${50 + Math.cos(angle) * r}%;`;
                    bubble.appendChild(dot);
                });

                // Count label
                const countEl = document.createElement('div');
                countEl.className = 'cluster-bubble-count';
                countEl.textContent = count === 1 ? '1' : `${count} videos`;
                bubble.appendChild(countEl);

                // Name label
                const label = document.createElement('div');
                label.className = 'cluster-bubble-label';
                label.innerHTML = `${escapeHtml(cluster.name)}<br><span style="font-size:8px;color:var(--text-tertiary)">${count > 1 ? `avg ${Math.round(cluster.avg_similarity * 100)}% similar` : 'no similar'}</span>`;
                bubble.appendChild(label);

                bubble.addEventListener('click', () => {
                    selectCluster(cluster.id);
                    flyToCluster(cluster);
                });

                bubbles.appendChild(bubble);
            });

            // Render similarity lines between clusters
            if (lines && clusterData.length > 1) {
                lines.innerHTML = '';
                for (let i = 0; i < clusterData.length; i++) {
                    for (let j = i + 1; j < clusterData.length; j++) {
                        const a = clusterData[i], b = clusterData[j];
                        // Compute inter-cluster similarity from centroids
                        const sim = cosineSim(a.centroid, b.centroid);
                        if (sim > 0.5) {
                            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                            line.setAttribute('x1', (a.position.x * 100) + '%');
                            line.setAttribute('y1', (a.position.y * 100) + '%');
                            line.setAttribute('x2', (b.position.x * 100) + '%');
                            line.setAttribute('y2', (b.position.y * 100) + '%');
                            const alpha = Math.max(0.05, (sim - 0.5) * 0.4);
                            line.setAttribute('stroke', `rgba(0,220,130,${alpha})`);
                            line.setAttribute('stroke-width', '1');
                            line.setAttribute('stroke-dasharray', '4,4');
                            lines.appendChild(line);
                        }
                    }
                }
            }

            // Auto-select first cluster
            if (clusterData.length > 0) {
                selectCluster(clusterData[0].id);
            }
        })
        .catch(err => {
            console.error('Failed to load clusters:', err);
            bubbles.innerHTML = '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--text-tertiary);font-size:12px">Failed to load videos.</div>';
        });
}

// Cosine similarity helper (client-side, for similarity lines)
function cosineSim(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}
```

- [ ] **Step 4: Add navigation event listeners**

Add after the cluster functions (before `compareCheckedVideos`). **Important:** These are NOT IIFEs — they are named functions called from `loadCompareVideos()` after the DOM elements exist. The compare page starts as `display:none`, so the elements don't exist at parse time.

```javascript
// ── Cluster viewport navigation ──
let clusterNavInitialized = false;
function initClusterNav() {
    if (clusterNavInitialized) return;
    const vp = document.getElementById('clusterViewport');
    if (!vp) return;
    clusterNavInitialized = true;

    // Scroll to zoom toward cursor
    vp.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = vp.getBoundingClientRect();
        const mx = e.clientX - rect.left - rect.width / 2;
        const my = e.clientY - rect.top - rect.height / 2;
        const oldZoom = clusterZoom;
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        clusterZoom = Math.max(CLUSTER_ZOOM_MIN, Math.min(CLUSTER_ZOOM_MAX, clusterZoom * delta));
        const scale = clusterZoom / oldZoom;
        clusterPanX = mx - scale * (mx - clusterPanX);
        clusterPanY = my - scale * (my - clusterPanY);
        applyClusterTransform();
    }, { passive: false });

    // Drag to pan
    vp.addEventListener('mousedown', (e) => {
        if (e.target.closest('.cluster-bubble') || e.target.closest('.cluster-panel') ||
            e.target.closest('.cluster-search-bar') || e.target.closest('.cluster-zoom-controls')) return;
        clusterDragging = true;
        clusterDragStartX = e.clientX;
        clusterDragStartY = e.clientY;
        vp.classList.add('dragging');
    });
    document.addEventListener('mousemove', (e) => {
        if (!clusterDragging) return;
        clusterPanX += e.clientX - clusterDragStartX;
        clusterPanY += e.clientY - clusterDragStartY;
        clusterDragStartX = e.clientX;
        clusterDragStartY = e.clientY;
        applyClusterTransform();
    });
    document.addEventListener('mouseup', () => {
        clusterDragging = false;
        document.getElementById('clusterViewport')?.classList.remove('dragging');
    });

    // Zoom buttons
    document.getElementById('clusterZoomIn')?.addEventListener('click', () => {
        clusterZoom = Math.min(CLUSTER_ZOOM_MAX, clusterZoom * 1.2);
        applyClusterTransform();
    });
    document.getElementById('clusterZoomOut')?.addEventListener('click', () => {
        clusterZoom = Math.max(CLUSTER_ZOOM_MIN, clusterZoom / 1.2);
        applyClusterTransform();
    });
    document.getElementById('clusterZoomReset')?.addEventListener('click', () => {
        const scene = document.getElementById('clusterScene');
        if (scene) scene.classList.add('animating');
        clusterZoom = 1.0; clusterPanX = 0; clusterPanY = 0;
        applyClusterTransform();
        setTimeout(() => scene?.classList.remove('animating'), 650);
    });
}
```

- [ ] **Step 5: Add search bar logic**

Add after the navigation function:

```javascript
// ── Cluster search ──
let clusterSearchInitialized = false;
function initClusterSearch() {
    if (clusterSearchInitialized) return;
    const input = document.getElementById('clusterSearchInput');
    const goBtn = document.getElementById('clusterSearchGo');
    const results = document.getElementById('clusterSearchResults');
    if (!input || !goBtn || !results) return;
    clusterSearchInitialized = true;

    function clearSearch() {
        input.value = '';
        results.innerHTML = '';
        document.querySelectorAll('.cluster-bubble').forEach(b => {
            b.classList.remove('search-hit', 'search-partial', 'search-dimmed');
        });
        const scene = document.getElementById('clusterScene');
        if (scene) scene.classList.add('animating');
        clusterZoom = 1.0; clusterPanX = 0; clusterPanY = 0;
        applyClusterTransform();
        setTimeout(() => scene?.classList.remove('animating'), 650);
    }

    async function doSearch() {
        const q = input.value.trim();
        results.innerHTML = '';
        document.querySelectorAll('.cluster-bubble').forEach(b => {
            b.classList.remove('search-hit', 'search-partial', 'search-dimmed');
        });
        if (!q || clusterData.length === 0) return;

        // Build centroids map
        const centroids = {};
        clusterData.forEach(c => { centroids[c.id] = c.centroid; });

        try {
            const resp = await fetch('/api/clusters/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: q, centroids })
            });
            const data = await resp.json();
            const scores = data.scores || {};

            const scored = clusterData.map(c => ({
                cluster: c,
                score: scores[c.id] || 0
            })).sort((a, b) => b.score - a.score);

            const hits = scored.filter(s => s.score >= 0.7);
            const partials = scored.filter(s => s.score >= 0.4 && s.score < 0.7);
            const misses = scored.filter(s => s.score < 0.4);

            if (hits.length === 0 && partials.length === 0) {
                results.innerHTML = '<span style="font-size:10px;color:var(--text-tertiary)">No matching clusters</span>';
                return;
            }

            // Apply visual states
            hits.forEach(s => {
                document.querySelector(`.cluster-bubble[data-cluster-id="${s.cluster.id}"]`)?.classList.add('search-hit');
            });
            partials.forEach(s => {
                document.querySelector(`.cluster-bubble[data-cluster-id="${s.cluster.id}"]`)?.classList.add('search-partial');
            });
            misses.forEach(s => {
                document.querySelector(`.cluster-bubble[data-cluster-id="${s.cluster.id}"]`)?.classList.add('search-dimmed');
            });

            // Result chips
            [...hits, ...partials].forEach(s => {
                const chip = document.createElement('span');
                chip.className = 'cluster-search-chip ' + (s.score >= 0.7 ? 'match' : 'partial');
                chip.textContent = `${s.cluster.name} ${Math.round(s.score * 100)}%`;
                chip.addEventListener('click', () => {
                    selectCluster(s.cluster.id);
                    flyToCluster(s.cluster);
                });
                results.appendChild(chip);
            });

            const clearBtn = document.createElement('span');
            clearBtn.className = 'cluster-search-clear';
            clearBtn.textContent = 'clear';
            clearBtn.addEventListener('click', clearSearch);
            results.appendChild(clearBtn);

            // Fly camera to results
            flyToFitClusters([...hits, ...partials].map(s => s.cluster));
        } catch (err) {
            console.error('Cluster search failed:', err);
            results.innerHTML = '<span style="font-size:10px;color:var(--error)">Search failed</span>';
        }
    }

    goBtn.addEventListener('click', doSearch);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doSearch();
        if (e.key === 'Escape') clearSearch();
    });
}
```

Then add calls to both init functions inside `loadCompareVideos()`, at the very beginning of the function body (after the DOM element lookups):

```javascript
// Inside loadCompareVideos(), after the element lookups:
initClusterNav();
initClusterSearch();
```

- [ ] **Step 6: Test manually in browser**

1. Start the app: `uvicorn app:app --reload --port 8000`
2. Open `http://localhost:8000` in browser
3. Navigate to Compare page
4. Verify:
   - Cluster bubbles render with correct sizes and positions
   - Drag to pan works
   - Scroll to zoom works (zooms toward cursor)
   - +/−/reset buttons work
   - Click a cluster → panel opens with video list
   - Check videos in panel → Compare button enables
   - Click Compare → NLE timeline renders
   - Search bar → type "test" → clusters highlight, camera flies

- [ ] **Step 6b: Remove old hidden `#compareVideoGrid` div**

Now that `updateRefBadge()` targets `#clusterPanelVideos` (Step 2), the old hidden `#compareVideoGrid` is dead HTML. Remove it from `static/index.html`.

- [ ] **Step 7: Commit**

```bash
git add static/index.html
git commit -m "feat: implement 3D cluster rendering, pan/zoom navigation, selection panel, and semantic search"
```

---

## Chunk 3: Polish + Deploy

### Task 6: Gentle float animation + visual polish

**Files:**
- Modify: `static/index.html` — JS section

Add the gentle floating animation from the mockup to the cluster bubbles, and wire up the SVG similarity lines to update positions during float.

- [ ] **Step 1: Add float animation after cluster rendering**

**IMPORTANT: Placement.** This code must go inside the `.then(data => { ... })` callback of `loadCompareVideos()`, specifically after the `clusterData.forEach(...)` loop that calls `bubbles.appendChild(bubble)` and before the similarity lines section. It runs inside the async fetch resolution — if placed outside `.then()`, the bubbles won't exist yet and the animation silently does nothing.

```javascript
// Gentle float animation
(function animateFloat() {
    const allBubbles = document.querySelectorAll('.cluster-bubble');
    allBubbles.forEach((b, i) => {
        const baseLeft = parseFloat(b.style.left);
        const baseTop = parseFloat(b.style.top);
        const speed = 0.0004 + i * 0.00015;
        const amp = 0.3 + i * 0.15; // subtle percentage drift
        const phase = i * 1.3;

        function tick(t) {
            const dy = Math.sin(t * speed + phase) * amp;
            const dx = Math.cos(t * speed * 0.7 + phase) * (amp * 0.6);
            b.style.top = (baseTop + dy) + '%';
            b.style.left = (baseLeft + dx) + '%';
            requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    });
})();
```

- [ ] **Step 2: Commit**

```bash
git add static/index.html
git commit -m "feat: add gentle float animation to cluster bubbles"
```

---

### Task 7: Test end-to-end and deploy to staging

**Files:** No file changes — testing and deployment only.

**Context:** Staging App Runner deploys from `feature/nav-redesign-compare-upload` branch. ARN: `arn:aws:apprunner:us-east-1:026090552520:service/video-search-staging/f5e755cb7c304fe4ba1b18a0b6154992`. URL: `i8gfmfket3.us-east-1.awsapprunner.com`. **NEVER merge to main or deploy production without explicit user approval.**

- [ ] **Step 1: Run clustering tests**

```bash
cd /Users/bpenven/Documents/Code/multi-modal-video-search
python -m pytest tests/test_clustering.py -v
```

Expected: All tests PASS

- [ ] **Step 2: Start app locally and test**

```bash
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000`, go to Compare page, and verify:
- Clusters render with correct grouping
- Pan/zoom/fly-to all work
- Cluster click → panel with videos
- Check 2+ videos → Compare enables
- Click Compare → NLE timeline works
- Search bar → type query → clusters highlight and camera flies
- Search clear → resets
- Unplayable files show warning
- Cross-cluster selection: check video in Cluster A, click Cluster B, verify "N selected from other clusters" indicator appears
- Float animation: bubbles gently drift after loading

- [ ] **Step 3: Push to feature branch**

```bash
git push origin feature/nav-redesign-compare-upload
```

- [ ] **Step 4: Deploy to staging**

```bash
aws apprunner start-deployment \
  --service-arn "arn:aws:apprunner:us-east-1:026090552520:service/video-search-staging/f5e755cb7c304fe4ba1b18a0b6154992" \
  --region us-east-1
```

Wait for deployment to complete (~5 min).

- [ ] **Step 5: Verify deployment status before testing**

```bash
aws apprunner describe-service \
  --service-arn "arn:aws:apprunner:us-east-1:026090552520:service/video-search-staging/f5e755cb7c304fe4ba1b18a0b6154992" \
  --region us-east-1 \
  --query "Service.Status" --output text
```

Expected: `RUNNING` (not `OPERATION_IN_PROGRESS`). If still in progress, wait and re-check.

- [ ] **Step 6: Verify on staging**

Open `https://i8gfmfket3.us-east-1.awsapprunner.com/` and repeat the manual test checklist from Step 2.
