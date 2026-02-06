# MongoDB Atlas Setup Guide

This guide walks you through setting up MongoDB Atlas for the multi-modal video search system.

## Prerequisites

- MongoDB Atlas account (free tier available)
- M10+ cluster tier (required for 4 vector indexes)

## Step 1: Create MongoDB Atlas Cluster

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a new project or use existing
3. Click "Build a Database"
4. Select cluster tier:
   - **M10 or higher** (required for 4 vector indexes)
   - Region: Same as AWS region (us-east-1 recommended)
5. Create cluster (takes ~5-10 minutes)

## Step 2: Configure Network Access

1. Go to Network Access → Add IP Address
2. Add your IP addresses:
   - Your local IP (for development)
   - AWS Lambda IPs (or 0.0.0.0/0 for all IPs - less secure)

## Step 3: Create Database User

1. Go to Database Access → Add New Database User
2. Create user with:
   - Username: `video_search_user` (or your choice)
   - Password: Generate secure password
   - Role: `Atlas admin` or `Read and write to any database`
3. Save credentials securely

## Step 4: Get Connection String

1. Go to Database → Connect → Drivers
2. Select Driver: Python, Version: 3.11 or later
3. Copy connection string:
   ```
   mongodb+srv://video_search_user:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
4. Replace `<password>` with your actual password
5. Add to `.env` file:
   ```bash
   MONGODB_URI=mongodb+srv://video_search_user:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   MONGODB_DATABASE=video_search
   ```

## Step 5: Create Vector Search Indexes

### Option A: Using pymongo (Programmatic)

Run the provided script:

```bash
python scripts/create_mongodb_indexes.py
```

This creates 4 vector indexes:
1. `unified_embeddings_vector_index` (single-index mode)
2. `visual_embeddings_vector_index` (multi-index mode)
3. `audio_embeddings_vector_index` (multi-index mode)
4. `transcription_embeddings_vector_index` (multi-index mode)

### Option B: Using Atlas UI (Manual)

1. Go to your cluster → Search tab
2. Click "Create Search Index"
3. Select "JSON Editor"
4. Create index with these settings:

**Unified Index** (for single-index mode):
```json
{
  "name": "unified_embeddings_vector_index",
  "type": "vectorSearch",
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

**Visual Index** (for multi-index mode):
```json
{
  "name": "visual_embeddings_vector_index",
  "type": "vectorSearch",
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

**Audio Index** (for multi-index mode):
```json
{
  "name": "audio_embeddings_vector_index",
  "type": "vectorSearch",
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

**Transcription Index** (for multi-index mode):
```json
{
  "name": "transcription_embeddings_vector_index",
  "type": "vectorSearch",
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

5. Wait for indexes to build (status: PENDING → READY, ~5-10 minutes each)

## Step 6: Verify Setup

Run the verification script:

```bash
python scripts/verify_mongodb_setup.py
```

This checks:
- ✅ Connection to MongoDB Atlas
- ✅ Database and collections exist
- ✅ Vector indexes are READY
- ✅ Can query vector indexes

## Troubleshooting

### Connection Timeout
- Check network access whitelist
- Verify connection string is correct
- Ensure password doesn't contain special characters (or URL-encode them)

### Vector Index Creation Failed
- Ensure cluster tier is M10+ (M0/M2/M5 don't support vector search)
- Check Atlas version is 6.0.11+ or 7.0.2+
- Verify field paths match your document structure

### Slow Queries
- Increase cluster tier (M10 → M20 → M30)
- Monitor query performance in Atlas UI
- Consider sharding for large datasets (>10M documents)

## Cost Optimization

**M10 Cluster Costs** (~$57/month):
- 2GB RAM, 10GB storage
- Supports 4 vector indexes
- Good for development and small production

**Recommendations:**
- Use M10 for development
- M20+ for production with >1M videos
- Enable auto-scaling for storage
- Set up data expiration policies

## Next Steps

1. Run video processing Lambda to populate data
2. Test search queries in the web UI
3. Monitor query performance in Atlas UI
4. Scale cluster as needed
