# MongoDB Atlas Setup Guide

This guide walks through setting up MongoDB Atlas for the multi-vector video search pipeline.

## Architecture Overview

The pipeline uses a **single collection** with a `modality_type` field for filtering:

| Collection | Purpose |
|------------|---------|
| `video_embeddings` | All embeddings (visual, audio, transcription) with modality_type filter |

**Single-collection approach benefits:**
- Pre-filter by `modality_type` to search specific modalities
- Search all modalities in one query without merging results
- Simpler index management
- Flexible fusion strategies (weighted or anchor-based)

---

## Step 1: Create MongoDB Atlas Cluster

1. Log in to [MongoDB Atlas](https://cloud.mongodb.com/)
2. Create a new cluster in **us-east-1** region (to minimize latency with AWS Bedrock)
3. Select **M10** or higher tier for vector search support
4. Name your cluster (e.g., `video-search-cluster`)

---

## Step 2: Configure Network Access

1. Go to **Network Access** in the Atlas sidebar
2. Click **Add IP Address**
3. For Lambda access, you have two options:

   **Option A: VPC Peering (Recommended for Production)**
   - Set up VPC peering between your AWS VPC and MongoDB Atlas
   - Add the VPC CIDR to the IP Access List

   **Option B: Allow All IPs (Development Only)**
   - Add `0.0.0.0/0` to allow access from any IP
   - **Warning**: Only use this for development/testing

---

## Step 3: Create Database User

1. Go to **Database Access** in the Atlas sidebar
2. Click **Add New Database User**
3. Create a user with the following settings:
   - Authentication Method: **Password**
   - Username: `video_search_user` (or your preferred name)
   - Password: Generate a secure password
   - Database User Privileges: **Read and write to any database**
4. Save the credentials securely

---

## Step 4: Get Connection String

1. Go to your cluster and click **Connect**
2. Select **Connect your application**
3. Select **Python** driver version **3.12 or later**
4. Copy the connection string, it will look like:

```
mongodb+srv://video_search_user:<password>@video-search-cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority
```

5. Replace `<password>` with your actual password

---

## Step 5: Create Database and Collection

Connect to your cluster using `mongosh` or the Atlas UI:

```bash
# Using mongosh
mongosh "mongodb+srv://video_search_user:<password>@video-search-cluster.xxxxx.mongodb.net/"
```

Create the database and collection:

```javascript
// Switch to the video_search database
use video_search

// Create the single collection with schema validation
db.createCollection("video_embeddings", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["video_id", "segment_id", "modality_type", "s3_uri", "embedding", "start_time", "end_time"],
      properties: {
        video_id: {
          bsonType: "string",
          description: "Unique video identifier"
        },
        segment_id: {
          bsonType: "int",
          description: "Segment index within video"
        },
        modality_type: {
          enum: ["visual", "audio", "transcription"],
          description: "Type of embedding modality"
        },
        s3_uri: {
          bsonType: "string",
          description: "S3 URI of source video"
        },
        embedding: {
          bsonType: "array",
          description: "512-dimensional embedding vector"
        },
        start_time: {
          bsonType: "double",
          description: "Segment start time (seconds)"
        },
        end_time: {
          bsonType: "double",
          description: "Segment end time (seconds)"
        },
        created_at: {
          bsonType: "date",
          description: "Document creation timestamp"
        }
      }
    }
  }
})
```

---

## Step 6: Create HNSW Vector Search Index

Vector search indices must be created through the **Atlas UI** or **Atlas Search API**.

### Option A: Atlas UI (Recommended)

1. Go to your cluster in Atlas
2. Click **Atlas Search** tab
3. Click **Create Search Index**
4. Select **JSON Editor**
5. Configure the index:

**Index Name:** `video_embeddings_vector_index`

**Collection:** `video_embeddings`

**JSON Definition:**

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 512,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "modality_type"
    },
    {
      "type": "filter",
      "path": "video_id"
    },
    {
      "type": "filter",
      "path": "segment_id"
    }
  ]
}
```

6. Click **Create Search Index**
7. Wait for the index status to become **Active** (may take a few minutes)

### Option B: Atlas Admin API

Create the index programmatically using the Atlas Admin API:

```bash
# Set your Atlas credentials
ATLAS_PUBLIC_KEY="your-public-key"
ATLAS_PRIVATE_KEY="your-private-key"
PROJECT_ID="your-project-id"
CLUSTER_NAME="video-search-cluster"

# Create vector search index
curl -X POST \
  "https://cloud.mongodb.com/api/atlas/v2/groups/${PROJECT_ID}/clusters/${CLUSTER_NAME}/fts/indexes" \
  --header "Content-Type: application/json" \
  --digest --user "${ATLAS_PUBLIC_KEY}:${ATLAS_PRIVATE_KEY}" \
  --data '{
    "name": "video_embeddings_vector_index",
    "database": "video_search",
    "collectionName": "video_embeddings",
    "type": "vectorSearch",
    "definition": {
      "fields": [
        {
          "type": "vector",
          "path": "embedding",
          "numDimensions": 512,
          "similarity": "cosine"
        },
        {
          "type": "filter",
          "path": "modality_type"
        },
        {
          "type": "filter",
          "path": "video_id"
        }
      ]
    }
  }'
```

---

## Step 7: Create Standard Indices for Filtering

Create compound indices for efficient filtering and lookups:

```javascript
// In mongosh, connected to your cluster
use video_search

// Unique constraint on video + segment + modality combination
db.video_embeddings.createIndex(
  { "video_id": 1, "segment_id": 1, "modality_type": 1 },
  { unique: true }
)

// Index for filtering by video
db.video_embeddings.createIndex({ "video_id": 1 })

// Index for filtering by modality
db.video_embeddings.createIndex({ "modality_type": 1 })

// Index for time-based queries
db.video_embeddings.createIndex({ "created_at": -1 })

// Compound index for common query patterns
db.video_embeddings.createIndex({ "video_id": 1, "modality_type": 1 })
```

---

## Step 8: Verify Setup

Run these commands to verify your setup:

```javascript
// Check collection exists
show collections

// Check indices
db.video_embeddings.getIndexes()

// Check document count (should be 0 initially)
db.video_embeddings.countDocuments()

// Check counts by modality (after inserting data)
db.video_embeddings.aggregate([
  { $group: { _id: "$modality_type", count: { $sum: 1 } } }
])
```

To verify vector search index, go to **Atlas Search** tab in the UI and confirm the index shows status **Active**.

---

## Schema Reference

### Document Schema

Each embedding document follows this schema:

| Field | Type | Description |
|-------|------|-------------|
| `_id` | ObjectId | Auto-generated document ID |
| `video_id` | String | Unique identifier for the video |
| `segment_id` | Integer | Segment index (0-based) |
| `modality_type` | String | One of: "visual", "audio", "transcription" |
| `s3_uri` | String | S3 URI of the source video |
| `embedding` | Array[512] | 512-dimensional embedding vector |
| `start_time` | Double | Segment start time in seconds |
| `end_time` | Double | Segment end time in seconds |
| `created_at` | Date | Timestamp when document was created |

### Example Documents

**Visual embedding:**
```json
{
  "_id": ObjectId("..."),
  "video_id": "a1b2c3d4e5f6g7h8",
  "segment_id": 0,
  "modality_type": "visual",
  "s3_uri": "s3://your-media-bucket-name/proxies/example.mp4",
  "embedding": [0.0234, -0.0156, 0.0891, ...],
  "start_time": 0.0,
  "end_time": 5.2,
  "created_at": ISODate("2024-01-15T10:30:00Z")
}
```

**Audio embedding (same segment):**
```json
{
  "_id": ObjectId("..."),
  "video_id": "a1b2c3d4e5f6g7h8",
  "segment_id": 0,
  "modality_type": "audio",
  "s3_uri": "s3://your-media-bucket-name/proxies/example.mp4",
  "embedding": [0.0567, -0.0234, 0.0123, ...],
  "start_time": 0.0,
  "end_time": 5.2,
  "created_at": ISODate("2024-01-15T10:30:00Z")
}
```

**Transcription embedding (same segment):**
```json
{
  "_id": ObjectId("..."),
  "video_id": "a1b2c3d4e5f6g7h8",
  "segment_id": 0,
  "modality_type": "transcription",
  "s3_uri": "s3://your-media-bucket-name/proxies/example.mp4",
  "embedding": [0.0891, -0.0345, 0.0678, ...],
  "start_time": 0.0,
  "end_time": 5.2,
  "created_at": ISODate("2024-01-15T10:30:00Z")
}
```

---

## Query Examples

### Search Single Modality (Pre-filtered)

Search only transcription embeddings for dialogue:

```javascript
db.video_embeddings.aggregate([
  {
    $vectorSearch: {
      index: "video_embeddings_vector_index",
      path: "embedding",
      queryVector: [/* 512-dim query embedding */],
      numCandidates: 100,
      limit: 10,
      filter: { modality_type: "transcription" }
    }
  },
  {
    $project: {
      video_id: 1,
      segment_id: 1,
      modality_type: 1,
      start_time: 1,
      end_time: 1,
      score: { $meta: "vectorSearchScore" }
    }
  }
])
```

### Search All Modalities (for Fusion)

Search without modality filter, then fuse in application:

```javascript
db.video_embeddings.aggregate([
  {
    $vectorSearch: {
      index: "video_embeddings_vector_index",
      path: "embedding",
      queryVector: [/* 512-dim query embedding */],
      numCandidates: 150,
      limit: 50
      // No filter - returns all modalities
    }
  },
  {
    $project: {
      video_id: 1,
      segment_id: 1,
      modality_type: 1,
      start_time: 1,
      end_time: 1,
      score: { $meta: "vectorSearchScore" }
    }
  }
])
```

### Search Within Specific Video

```javascript
db.video_embeddings.aggregate([
  {
    $vectorSearch: {
      index: "video_embeddings_vector_index",
      path: "embedding",
      queryVector: [/* 512-dim query embedding */],
      numCandidates: 100,
      limit: 10,
      filter: {
        video_id: "a1b2c3d4e5f6g7h8",
        modality_type: "visual"
      }
    }
  }
])
```

---

## Troubleshooting

### Vector Search Index Not Working

- Ensure the index status is **Active** (can take a few minutes after creation)
- Verify the `numDimensions` matches exactly (512)
- Check that the `path` is set to `embedding`
- Confirm filter fields are included in the index definition

### Connection Issues from Lambda

- Verify the MongoDB Atlas IP Access List includes Lambda's IP range
- For production, use VPC peering or AWS PrivateLink
- Check the connection string format is correct

### Slow Query Performance

- Increase `numCandidates` in vector search (higher = more accurate but slower)
- Use pre-filtering with `modality_type` when you know which modality to search
- Use `video_id` filter when searching within a specific video
- Monitor Atlas metrics for index utilization

### Filter Not Working

- Ensure filter fields (`modality_type`, `video_id`) are defined in the vector index
- Check that filter values match exactly (case-sensitive)

---

## Next Steps

After completing this setup:

1. Set the `MONGODB_URI` environment variable
2. Run the Lambda deployment script: `./scripts/deploy.sh`
3. Test the pipeline with a sample video
4. Run queries using the fusion testing script:
   ```bash
   # Fusion search (all modalities)
   python src/query_fusion.py "a person walking"

   # Single modality search
   python src/query_fusion.py "someone talking about revenue" --single-modality transcription
   ```
