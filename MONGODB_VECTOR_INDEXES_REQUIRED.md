# MongoDB Atlas Vector Search Indexes - REQUIRED

## Issue
MongoDB search returns 0 results because vector search indexes are missing.

## Required Indexes

Create these 4 vector search indexes in MongoDB Atlas:

### 1. unified-embeddings collection
**Index Name:** `unified_embeddings_vector_index`
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
    }
  ]
}
```

### 2. visual_embeddings collection
**Index Name:** `visual_embeddings_vector_index`
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
      "path": "video_id"
    }
  ]
}
```

### 3. audio_embeddings collection
**Index Name:** `audio_embeddings_vector_index`
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
      "path": "video_id"
    }
  ]
}
```

### 4. transcription_embeddings collection
**Index Name:** `transcription_embeddings_vector_index`
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
      "path": "video_id"
    }
  ]
}
```

## How to Create (MongoDB Atlas UI)

1. Go to MongoDB Atlas: https://cloud.mongodb.com
2. Select your cluster (Cluster0)
3. Click "Search" in the left sidebar
4. Click "Create Search Index"
5. Select "JSON Editor"
6. Choose "Atlas Vector Search"
7. Select database: `video_search`
8. Select collection (e.g., `unified-embeddings`)
9. Index Name: Enter the exact name from above
10. Paste the JSON configuration
11. Click "Create Search Index"
12. Repeat for all 4 collections

## Verification

After creating indexes, run this test:

```bash
python3 << 'EOF'
from pymongo import MongoClient
client = MongoClient("YOUR_MONGODB_URI")
db = client["video_search"]

for coll_name in ["unified-embeddings", "visual_embeddings", "audio_embeddings", "transcription_embeddings"]:
    indexes = list(db[coll_name].list_search_indexes())
    print(f"{coll_name}: {len(indexes)} search indexes")
    for idx in indexes:
        print(f"  - {idx['name']}")
EOF
```

Expected output:
```
unified-embeddings: 1 search indexes
  - unified_embeddings_vector_index
visual_embeddings: 1 search indexes
  - visual_embeddings_vector_index
audio_embeddings: 1 search indexes
  - audio_embeddings_vector_index
transcription_embeddings: 1 search indexes
  - transcription_embeddings_vector_index
```
