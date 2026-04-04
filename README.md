# Multi-Vector Video Search Pipeline

A video semantic search and comparison pipeline built with AWS Bedrock Marengo 3.0 and **MongoDB Atlas** multi-collection vector storage.

---

## Table of Contents

- [Multi-Vector Search Architecture](#multi-vector-search-architecture)
- [Fusion Methods](#fusion-methods)
  - [Reciprocal Rank Fusion (RRF)](#1-reciprocal-rank-fusion-rrf)
  - [Weighted Score Fusion](#2-weighted-score-fusion)
  - [Intent-Based Dynamic Routing](#3-intent-based-dynamic-routing)
- [LLM Query Decomposition](#llm-query-decomposition)
- [Modality Weight Configurations](#modality-weight-configurations)
- [Compare Mode](#compare-mode)
- [Architecture Overview](#architecture-overview)
- [Search UI Features](#search-ui-features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation & Deployment](#installation--deployment)
- [MongoDB Schema](#mongodb-schema)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)

---

## Multi-Vector Search Architecture

This system uses a **multi-vector retrieval architecture** where each video segment stores **three separate embedding vectors** (visual, audio, transcription) instead of one combined embedding.

**Why separate embeddings?** This allows you to adjust the importance of each modality at search time without re-indexing. You can search the same data with different strategies depending on your query.

**Storage Architecture:**
```
Video Segment → Three 512d Embeddings:
  ├─ Visual Embedding      (visual content, scenes, actions)
  ├─ Audio Embedding       (sounds, music, ambient audio)
  └─ Transcription Embedding (spoken words, dialogue)
```

Each modality is stored in its own MongoDB collection with a dedicated vector index.

**Advantages:**
- Preserves modality-specific signal fidelity
- Transparent, modality-level debuggability
- Change weights without re-indexing
- Supports modality-specific optimization
- Foundation for adaptive architectures

**Drawbacks:**
- 3x storage footprint vs single fused embedding
- 3 vector searches per query
- More complex infrastructure

---

## Fusion Methods

Three methods for combining multi-vector search results:

### 1. Reciprocal Rank Fusion (RRF)

**Formula:**
```
score(d) = Σ w_m / (k + rank_m(d))

Where:
  w_m = modality weight
  k = 60 (standard RRF constant)
  rank_m(d) = rank of document d in modality m
```

**Implementation:** `search_client.py`

**Characteristics:**
- Robust to score distribution differences
- Emphasizes agreement between modalities
- Standard approach (used by Elasticsearch, etc.)
- Better for diverse query distributions

**Default Weights:**
```python
{
  "visual": 0.8,      # 80% weight on visual ranking
  "audio": 0.1,       # 10% weight on audio ranking
  "transcription": 0.05  # 5% weight on transcription ranking
}
```

**API Usage:**
```python
results = client.search(
    query="person running in park",
    fusion_method="rrf",
    weights={"visual": 0.8, "audio": 0.1, "transcription": 0.05}
)
```

---

### 2. Weighted Score Fusion

**Formula:**
```
score(s) = Σ w_m · sim(Q_m, E_m(s))

Where:
  w_m = modality weight
  sim() = cosine similarity
  Q_m = query embedding for modality m
  E_m(s) = segment embedding for modality m
```

**Implementation:** `search_client.py`

**Characteristics:**
- Direct score combination
- Simpler than RRF
- Sensitive to score distributions
- Works well with normalized scores

**Default Weights:**
```python
{
  "visual": 0.8,
  "audio": 0.1,
  "transcription": 0.1
}
```

**API Usage:**
```python
results = client.search(
    query="person running in park",
    fusion_method="weighted",
    weights={"visual": 0.8, "audio": 0.1, "transcription": 0.1}
)
```

---

### 3. Intent-Based Dynamic Routing

**Implementation:** Uses embedding similarity to anchor prompts to automatically compute weights.

**How It Works:**
1. Pre-compute anchor embeddings for each modality (at startup)
2. For each query, compute cosine similarity to each anchor
3. Apply softmax with temperature to get normalized weights

**Formula:**
```
(w_v, w_a, w_t) = softmax(α · sim(E_query, [E_AncV, E_AncA, E_AncT]))

Where:
  α = temperature (default: 10.0)
  E_query = query embedding
  E_AncV/A/T = anchor embeddings for visual/audio/transcription
```

**Anchor Prompts:**
```python
VISUAL_ANCHOR = "What appears on screen: people, objects, scenes, actions,
                 clothing, colors, and visual composition of the video."

AUDIO_ANCHOR = "The non-speech audio in the video: music, sound effects,
                ambient sound, and other audio elements."

TRANSCRIPTION_ANCHOR = "The spoken words in the video: dialogue, narration,
                        speech, and what people say."
```

**Characteristics:**
- Query-adaptive - weights change per query
- Deterministic - same query = same weights
- Explainable - can inspect anchor similarities
- No training required - uses embedding space directly
- Fast iteration - update anchors without retraining

**API Usage:**
```python
response = client.search_dynamic(
    query="explosion with loud bang",
    temperature=10.0  # Higher = more uniform, lower = more decisive
)

print(f"Computed weights: {response['weights']}")
# Output: {"visual": 0.45, "audio": 0.42, "transcription": 0.13}

print(f"Anchor similarities: {response['similarities']}")
# Output: {"visual": 0.78, "audio": 0.75, "transcription": 0.45}
```

**Temperature Effects:**
| Temperature | Behavior | Example Weights (visual, audio, transcription) |
|-------------|----------|-----------------------------------------------|
| `α = 1.0` | Very decisive (sharp distribution) | 0.89, 0.08, 0.03 |
| `α = 10.0` (default) | Balanced adaptation | 0.45, 0.42, 0.13 |
| `α = 50.0` | Uniform (ignores differences) | 0.34, 0.33, 0.33 |

---

## LLM Query Decomposition

**Purpose:** Decompose complex natural language queries into modality-specific sub-queries for enhanced precision.

**Implementation:** `bedrock_client.py`

**How It Works:**
1. User provides a natural language query
2. Claude 3 Haiku decomposes it into three distinct queries:
   - **Visual query**: What appears on screen
   - **Audio query**: Non-speech sounds only
   - **Transcription query**: Spoken words and dialogue
3. Each sub-query gets its own embedding
4. Separate vector searches per modality using appropriate embeddings

**Example:**

**Input Query:**
```
"Ross says I take thee Rachel at a wedding"
```

**LLM Decomposition:**
```python
{
  "visual": "Ross at a wedding ceremony, wedding altar, formal attire",
  "audio": "wedding music, ceremony sounds, emotional atmosphere",
  "transcription": "Ross says I take thee Rachel"
}
```

**Model Configuration:**
- **Model:** Claude 3 Haiku (`anthropic.claude-3-haiku-20240307-v1:0`)
- **Temperature:** 0.3 (low for deterministic structured output)
- **Max Tokens:** 500

**API Usage:**
```python
# Enable decomposition with flag
results = client.search(
    query="Ross says I take thee Rachel at a wedding",
    fusion_method="rrf",
    decomposed_queries=client.bedrock.decompose_query(query)
)
```

**Web UI:** Enable "Use LLM Decomposition" toggle

**Characteristics:**
- Precision boost for complex multi-modal queries
- Extracts distinct signals from ambiguous queries
- Context-aware expansion - infers relevant elements
- Adds latency (~500ms for LLM call)
- Requires Bedrock access to Claude models

**Best For:**
- Complex queries spanning multiple modalities
- Queries where visual/audio/speech elements are intertwined
- When maximum precision is more important than latency

**Not Recommended For:**
- Simple single-modality queries ("red car")
- High-throughput/low-latency requirements
- Cost-sensitive applications (adds LLM inference cost)

---

## Modality Weight Configurations

### 1. Fixed Weights

**Method:** Manually set or statistically optimized weights applied to all queries.

**Default (Visual-Heavy):**
```python
VISUAL_WEIGHT = 0.8
AUDIO_WEIGHT = 0.1
TRANSCRIPTION_WEIGHT = 0.1
```

**Recommended Configurations by Use Case:**

| Use Case | Visual | Audio | Transcription | Example Query |
|----------|--------|-------|---------------|---------------|
| **Visual-Centric** | 0.80 | 0.10 | 0.10 | "person running", "red car crash" |
| **Dialogue-Focused** | 0.20 | 0.10 | 0.70 | "what did they say about revenue", "find where he mentions the deadline" |
| **Audio Events** | 0.30 | 0.60 | 0.10 | "explosion sound", "alarm ringing", "music playing" |
| **Balanced** | 0.40 | 0.30 | 0.30 | "wedding ceremony", "basketball game" |
| **Speech-Heavy + Visual** | 0.40 | 0.10 | 0.50 | "presenter showing slides", "interview about product" |

**Configuration Methods:**

**1. Environment Variables:**
```bash
export WEIGHT_VISUAL=0.8
export WEIGHT_AUDIO=0.1
export WEIGHT_TRANSCRIPTION=0.1
```

**2. API Parameters:**
```python
results = client.search(
    query="person laughing at joke",
    weights={"visual": 0.4, "audio": 0.3, "transcription": 0.3}
)
```

**3. Web UI Sliders:**
- Adjust visual/audio/transcription sliders in real-time
- Weights automatically normalize to sum to 1.0

### 2. Dynamic Routing with Anchors

**Method:** Automatically compute weights per query using anchor similarity.

See [Intent-Based Dynamic Routing](#3-intent-based-dynamic-routing) above for detailed explanation.

**Query-Specific Weight Examples:**

| Query | Visual | Audio | Transcription | Reasoning |
|-------|--------|-------|---------------|-----------|
| "person running in park" | 0.71 | 0.15 | 0.14 | Strong visual signal |
| "explosion with loud bang" | 0.45 | 0.42 | 0.13 | Visual + audio balanced |
| "he says I take thee Rachel" | 0.22 | 0.12 | 0.66 | Heavily speech-focused |
| "wedding ceremony music" | 0.38 | 0.47 | 0.15 | Audio-dominant |
| "red car crash" | 0.68 | 0.18 | 0.14 | Visual with some audio |

**API Usage:**
```python
response = client.search_dynamic(
    query="explosion with loud bang",
    temperature=10.0,
    limit=50
)

# Inspect computed weights
print(f"Query: {query}")
print(f"Visual weight: {response['weights']['visual']:.2f}")
print(f"Audio weight: {response['weights']['audio']:.2f}")
print(f"Transcription weight: {response['weights']['transcription']:.2f}")

# Results
for result in response['results']:
    print(f"Segment {result['segment_id']}: {result['fusion_score']:.3f}")
```

---

## Compare Mode

The compare feature performs segment-level diff between two or more videos using embedding similarity, designed for QC workflows and duplicate/version detection.

### Similarity Methods

Three methods are available for measuring segment similarity. Each catches a different class of difference:

| Method | Threshold | Catches | Best For |
|--------|-----------|---------|----------|
| **Cosine** | 0.95 | Structural edits (cuts, reorders, content swaps) | Detecting editorial changes |
| **L2 / Euclidean** | 0.85 | Magnitude changes (color grading, luma shifts, compression) | Detecting visual processing changes |
| **Combined** (default) | Union of both | Everything above | Recommended default -- highest recall |

**Combined** is the recommended default. Cosine catches structural edits that L2 misses, and L2 catches color/luma changes that cosine misses. Using both in union provides the best coverage.

### Difference Classification

Segments flagged as different are automatically classified:

| Classification | Description |
|----------------|-------------|
| `removed_content` | Segment exists in reference but not in compare target |
| `added_content` | Segment exists in compare target but not in reference |
| `significant_change` | Major structural or content difference |
| `color_luma_change` | Color grading or brightness change (L2-detected) |
| `compression_artifact` | Minor quality loss from re-encoding |
| `shot_reorder` | Same content found at a different timecode |

### VMAF Integration

Per-segment VMAF (Video Multi-Method Assessment Fusion) pixel-level analysis is available for segments flagged as different, providing a ground-truth quality metric alongside the embedding-based detection.

### Performance

- **83% recall** against the PerceptualHashing test matrix
- **88% recall** excluding standards conversion test cases (frame rate, resolution changes)

### Output Format

All timecodes in compare output use **SMPTE format** (HH:MM:SS:FF).

### API Endpoints

```
POST /api/compare/diff              - Pairwise diff between two videos
POST /api/compare/multi-diff        - Diff one reference against multiple targets
GET  /api/compare/similarity-methods - List available similarity methods and thresholds
```

**Example -- pairwise diff:**
```bash
curl "http://localhost:8000/api/compare/diff" \
  -H "Content-Type: application/json" \
  -d '{
    "reference_video_id": "video_a",
    "compare_video_id": "video_b",
    "similarity_method": "combined"
  }'
```

### UI

The compare view provides:
- Side-by-side segment timeline with thumbnails
- Similarity method toggle (Cosine / L2 / Combined)
- Difference segments highlighted with classification labels
- Filters: All / Time Offsets / Missing segments

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐
│   S3 Bucket     │     │  AWS Lambda      │
│   (Videos)      │────▶│  (Processing)    │
│                 │     │                  │
│ your-media-     │     │  ┌────────────┐  │
│ bucket-name/    │     │  │  Bedrock   │  │
│ input/          │     │  │  Marengo   │  │
└────────┬────────┘     │  │  3.0       │  │
         │              │  └────────────┘  │
    S3 Trigger          │                  │
    (automatic)         │  Embeddings:     │
                        │  - Visual (512d) │
                        │  - Audio (512d)  │
                        │  - Transcription │
                        │    (512d)        │
                        └─────────┬────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │       MongoDB Atlas            │
                  │       (Multi-Collection)       │
                  │                                │
                  │  ┌──────────────────────────┐  │
                  │  │  visual_embeddings       │  │
                  │  │  + HNSW vector index     │  │
                  │  ├──────────────────────────┤  │
                  │  │  audio_embeddings        │  │
                  │  │  + HNSW vector index     │  │
                  │  ├──────────────────────────┤  │
                  │  │  transcription_embeddings│  │
                  │  │  + HNSW vector index     │  │
                  │  ├──────────────────────────┤  │
                  │  │  video_fingerprints      │  │
                  │  │  (compare mode)          │  │
                  │  └──────────────────────────┘  │
                  └───────────────┬────────────────┘
                                  │
┌─────────────────┐     ┌────────┴─────────┐
│   CloudFront    │     │  AWS App Runner  │
│   (CDN)         │◀────│  (Search API)    │
│                 │     │                  │
│ Video streaming │     │  ┌────────────┐  │
│ + thumbnails    │     │  │  FastAPI   │  │
└─────────────────┘     │  │  + Multi   │  │
                        │  │    Fusion  │  │
                        │  │  + Dynamic │  │
                        │  │    Routing │  │
                        │  │  + Compare │  │
                        │  └────────────┘  │
                        │                  │
                        │  Fusion Methods: │
                        │  - RRF           │
                        │  - Weighted      │
                        │  - Dynamic       │
                        │                  │
                        │  Query Modes:    │
                        │  - LLM Decomp    │
                        │  - Single Query  │
                        │                  │
                        │  Compare Mode:   │
                        │  - Cosine        │
                        │  - L2            │
                        │  - Combined      │
                        └──────────────────┘
```

---

## Search UI Features

The web interface provides comprehensive search capabilities:

### Search Modes

**Multi-Vector Fusion:**
- **RRF** - Reciprocal Rank Fusion (rank-based, most robust)
- **Weighted** - Score-based fusion with adjustable weights
- **Dynamic** - Intent-based routing with automatic weight calculation

**Single Modality:**
- **Visual** - Visual content only (scenes, actions, objects)
- **Audio** - Audio/sound only (music, sound effects, ambient)
- **Speech** - Transcription/dialogue only (spoken words)

### Query Options

- **LLM Decomposition** - Enable/disable query decomposition with Claude
- **Modality Weights** - Real-time sliders for visual/audio/transcription weights
- **Temperature Control** - Adjust softmax temperature for dynamic routing (1-50)

### Result Card Layout

Each search result displays comprehensive match information:

```
┌─────────────────────────────┐
│ #1           85%     [VIS]  │  ← Rank, Confidence %, Dominant Modality
│                             │
│     [Video Thumbnail]       │
│                             │
│         0:30 - 1:15         │  ← Timestamp Range
└─────────────────────────────┘
  Video Title
  vis: 0.85  aud: 0.12  tra: 0.03  ← Individual Modality Scores
  ███████░░ ███░░░░░░░ █░░░░░░░░  ← Visual Score Bars
```

**Key Features:**
- **Ranking Badge** (#1, #2, #3...) - Shows result position
- **Confidence %** - Match confidence (0-100%)
- **Dominant Badge** - Which modality scored highest (VIS/AUD/TRA)
- **Modality Scores** - Detailed breakdown per embedding type
- **Score Visualization** - Visual bars showing relative strengths
- **20 Results per Page** - Focused, high-quality results

---

## Project Structure

```
multi-modal-video-search/
├── app.py                           # FastAPI web application (search + compare APIs)
├── src/
│   ├── lambda_function.py           # Lambda handler for video processing
│   ├── bedrock_client.py            # Bedrock Marengo client + LLM decomposition
│   ├── mongodb_client.py            # MongoDB multi-collection storage
│   ├── search_client.py             # Multi-vector search with all fusion methods
│   ├── compare_client.py            # Video comparison (segment diff, fingerprints)
│   └── clustering.py                # Agglomerative clustering for result grouping
├── static/
│   └── index.html                   # Search UI frontend (single-page app)
├── scripts/
│   ├── deploy.sh                    # AWS CLI deployment script
│   ├── setup_infrastructure.sh      # Automated AWS infrastructure setup
│   ├── create_mongodb_indexes.py    # Create MongoDB vector indexes
│   ├── backfill_fingerprints.py     # Backfill video fingerprints for compare mode
│   └── mongodb_setup.md             # MongoDB Atlas setup guide
├── qc_reports/                      # Generated comparison reports (per method)
│   ├── cosine/
│   ├── l2/
│   └── combined/
├── tests/
│   └── test_clustering.py           # Clustering unit tests
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container build for App Runner
├── apprunner.yaml                   # App Runner service configuration
├── .env.example                     # Environment variables template
└── README.md                        # This file
```

---

## Quick Start

### Prerequisites

Before starting, ensure you have:

- **AWS Account** with access to:
  - Bedrock (us-east-1 region for Marengo 3.0)
  - Lambda
  - S3
  - IAM (to create roles)
- **MongoDB Atlas** cluster (M10+ tier recommended for vector search indexes)
- **AWS CLI** installed and configured (`aws configure`)
- **Python 3.11+** installed
- **Git** for cloning the repository

### Installation & Deployment

**Two setup options:**

#### Option A: Automated Setup (Recommended)

Use the infrastructure setup script to deploy everything automatically.

**What it does:**
- Creates S3 buckets (media storage)
- Creates IAM roles with required permissions
- Sets up CloudFront distribution
- Deploys Lambda function
- Configures all environment variables

**Prerequisites:**
- AWS CLI configured with admin access
- MongoDB Atlas connection string (get from MongoDB Atlas UI)

**Steps:**

```bash
# 1. Clone repository
git clone https://github.com/your-username/multi-modal-video-search.git
cd multi-modal-video-search

# 2. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set these REQUIRED variables:
#   - MONGODB_URI (from MongoDB Atlas)
#   - MONGODB_DATABASE (default: video_search)
#   - AWS_ACCOUNT_ID (your 12-digit AWS account ID)
#   - S3_BUCKET (e.g., your-media-bucket-name)

# 4. Run automated setup
chmod +x scripts/setup_infrastructure.sh
./scripts/setup_infrastructure.sh
```

The script will prompt for confirmation before creating resources. Expected runtime: ~5-10 minutes.

---

#### Option B: Manual Setup

Follow these steps to set up each component individually.

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/your-username/multi-modal-video-search.git
cd multi-modal-video-search

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your credentials (see Environment Variables section below)
```

### 2. Setup MongoDB Atlas

Follow the detailed guide in [scripts/mongodb_setup.md](scripts/mongodb_setup.md):

1. Create a cluster (M10+ tier recommended for vector search indexes)
2. Create database user and get connection string
3. Create three modality collections: `visual_embeddings`, `audio_embeddings`, `transcription_embeddings`
4. Create vector indexes on each collection (use `scripts/create_mongodb_indexes.py`)
5. Whitelist IPs (or use 0.0.0.0/0 for testing)
6. Update `MONGODB_URI` and `MONGODB_DATABASE` in your `.env` file

**Create vector indexes:**
```bash
python scripts/create_mongodb_indexes.py
```

### 3. Deploy Lambda Function

The deployment script automates Lambda function creation, IAM role setup, and configuration.

```bash
# Make script executable (first time only)
chmod +x scripts/deploy.sh

# Set required environment variables
export MONGODB_URI="your_mongodb_connection_string_here"
export S3_BUCKET="your-media-bucket-name"
export CLOUDFRONT_DOMAIN="xxxxx.cloudfront.net"

# Run deployment script
./scripts/deploy.sh lambda
```

**What the script does:**
1. Validates AWS credentials and region
2. Creates IAM role with Bedrock + S3 + CloudWatch permissions
3. Packages Python dependencies into deployment zip
4. Creates/updates Lambda function with environment variables
5. Configures 15-minute timeout and 1024MB memory
6. Sets up CloudWatch logging

**Expected output:**
```
IAM role created: video-embedding-pipeline-role
Lambda function deployed: video-embedding-pipeline
Function size: 2.9 MB
Timeout: 900 seconds
Memory: 1024 MB
```

### 4. Run Search API Locally

```bash
# Start the FastAPI server
python app.py

# Open browser to http://localhost:8000
```

### 5. Process a Video

```bash
# Invoke Lambda
aws lambda invoke \
  --function-name video-embedding-pipeline \
  --region us-east-1 \
  --payload '{"s3_key": "input/sample.mp4", "bucket": "your-media-bucket-name"}' \
  --cli-binary-format raw-in-base64-out \
  response.json
```

### 6. Search Videos

**Via Web UI:** http://localhost:8000

**Via API:**
```bash
# Simple search with RRF fusion
curl "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "person running in park",
    "fusion_method": "rrf",
    "limit": 10
  }'

# Dynamic routing search
curl "http://localhost:8000/api/search/dynamic" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "explosion with loud bang",
    "temperature": 10.0,
    "limit": 10
  }'

# With LLM decomposition
curl "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Ross says I take thee Rachel at a wedding",
    "use_decomposition": true,
    "fusion_method": "rrf",
    "limit": 10
  }'
```

---

## MongoDB Schema

MongoDB uses **multi-collection mode** with one collection per modality plus a fingerprints collection for compare mode.

### Collections

| Collection | Purpose |
|------------|---------|
| `visual_embeddings` | Visual modality embeddings |
| `audio_embeddings` | Audio modality embeddings |
| `transcription_embeddings` | Transcription modality embeddings |
| `video_fingerprints` | Per-video mean embeddings for fast comparison |

### Document Schema

**Modality collections (visual_embeddings, audio_embeddings, transcription_embeddings):**
```json
{
  "_id": "ObjectId",
  "video_id": "string - unique video identifier",
  "segment_id": "int - segment index within video",
  "s3_uri": "string - s3://bucket/key",
  "embedding": "[float] - 512-dimensional vector",
  "start_time": "float - segment start (seconds)",
  "end_time": "float - segment end (seconds)",
  "created_at": "datetime - document creation time"
}
```
*No `modality_type` field needed -- the collection name implies the modality.*

**video_fingerprints collection:**
```json
{
  "_id": "ObjectId",
  "video_id": "string - unique video identifier",
  "visual_fingerprint": "[float] - 512d mean of visual embeddings",
  "audio_fingerprint": "[float] - 512d mean of audio embeddings",
  "transcription_fingerprint": "[float] - 512d mean of transcription embeddings",
  "segment_count": "int - number of segments",
  "total_duration": "float - video duration in seconds",
  "created_at": "datetime"
}
```

### Vector Index Definitions

Each modality collection has its own HNSW vector search index:

| Collection | Index Name | Vector Field | Dimensions | Similarity |
|------------|-----------|--------------|------------|------------|
| `visual_embeddings` | `visual_vector_index` | `embedding` | 512 | cosine |
| `audio_embeddings` | `audio_vector_index` | `embedding` | 512 | cosine |
| `transcription_embeddings` | `transcription_vector_index` | `embedding` | 512 | cosine |

Filter fields on each index: `video_id`.

---

## API Reference

### VideoSearchClient (search_client.py)

```python
from src.search_client import VideoSearchClient

client = VideoSearchClient(
    mongodb_uri="mongodb+srv://...",
    database_name="video_search"
)

# ============ RRF Fusion Search ============
results = client.search(
    query="person running",
    fusion_method="rrf",
    weights={"visual": 0.8, "audio": 0.1, "transcription": 0.1},
    limit=10
)

# ============ Weighted Fusion Search ============
results = client.search(
    query="person running",
    fusion_method="weighted",
    weights={"visual": 0.8, "audio": 0.1, "transcription": 0.1},
    limit=10
)

# ============ Dynamic Intent Routing ============
response = client.search_dynamic(
    query="explosion with loud bang",
    temperature=10.0,
    limit=10
)
print(f"Computed weights: {response['weights']}")
print(f"Anchor similarities: {response['similarities']}")

# ============ With LLM Query Decomposition ============
decomposed = client.bedrock.decompose_query("Ross says I take thee Rachel at a wedding")
print(f"Visual: {decomposed['visual']}")
print(f"Audio: {decomposed['audio']}")
print(f"Transcription: {decomposed['transcription']}")

results = client.search(
    query="Ross says I take thee Rachel at a wedding",
    decomposed_queries=decomposed,
    fusion_method="rrf",
    limit=10
)

# ============ Single Modality Search ============
results = client.search(
    query="person running",
    modalities=["visual"],  # Only search visual modality
    limit=10
)
```

### BedrockMarengoClient (bedrock_client.py)

```python
from src.bedrock_client import BedrockMarengoClient

client = BedrockMarengoClient(region="us-east-1")

# ============ Generate Video Embeddings ============
result = client.get_video_embeddings(
    bucket="your-media-bucket-name",
    s3_key="input/file.mp4",
    embedding_types=["visual", "audio", "transcription"]
)

# ============ Generate Query Embedding ============
query_result = client.get_text_query_embedding("a car driving fast")

# ============ LLM Query Decomposition ============
decomposed = client.decompose_query("Ross says I take thee Rachel at a wedding")
print(decomposed)
# {
#   "original_query": "Ross says I take thee Rachel at a wedding",
#   "visual": "Ross at a wedding ceremony, wedding altar, formal attire",
#   "audio": "wedding music, ceremony sounds, emotional atmosphere",
#   "transcription": "Ross says I take thee Rachel"
# }
```

### Compare API (app.py)

```bash
# ============ Pairwise Diff ============
curl "http://localhost:8000/api/compare/diff" \
  -H "Content-Type: application/json" \
  -d '{
    "reference_video_id": "video_a",
    "compare_video_id": "video_b",
    "similarity_method": "combined"
  }'

# ============ Multi-Diff (one reference vs many) ============
curl "http://localhost:8000/api/compare/multi-diff" \
  -H "Content-Type: application/json" \
  -d '{
    "reference_video_id": "video_a",
    "compare_video_ids": ["video_b", "video_c"],
    "similarity_method": "combined"
  }'

# ============ List Similarity Methods ============
curl "http://localhost:8000/api/compare/similarity-methods"
# Returns: ["cosine", "l2", "combined"]
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URI` | Required | MongoDB Atlas connection string |
| `MONGODB_DATABASE` | `video_search` | MongoDB database name |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `S3_BUCKET` | `your-media-bucket-name` | S3 bucket for videos |
| `CLOUDFRONT_DOMAIN` | `xxxxx.cloudfront.net` | CloudFront domain for streaming |
| `WEIGHT_VISUAL` | `0.8` | Default visual weight (fixed mode) |
| `WEIGHT_AUDIO` | `0.1` | Default audio weight (fixed mode) |
| `WEIGHT_TRANSCRIPTION` | `0.1` | Default transcription weight (fixed mode) |

---
