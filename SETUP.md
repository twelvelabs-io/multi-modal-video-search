# Setup Guide

## Overview

The system consists of:
1. **AWS Lambda** — Processes videos and generates embeddings
2. **MongoDB Atlas** — Stores embeddings (multi-collection: visual, audio, transcription)
3. **AWS S3** — Stores video files (originals + proxies)
4. **AWS CloudFront** — Delivers videos to the browser
5. **AWS App Runner** — Hosts the FastAPI search and comparison API
6. **Web UI** — Single-page app for search, compare, and clustering

## Prerequisites

- AWS account with access to Bedrock (us-east-1 for Marengo 3.0), Lambda, S3, App Runner
- AWS CLI installed and configured
- Python 3.11+
- MongoDB Atlas cluster (M10+ recommended for vector search indexes)

## 1. Clone and Configure

```bash
git clone https://github.com/twelvelabs-io/multi-modal-video-search.git
cd multi-modal-video-search

cp .env.example .env
```

Edit `.env`:
```bash
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?appName=<appName>
MONGODB_DATABASE=multi_modal_video_search
AWS_REGION=us-east-1
S3_BUCKET=your-media-bucket-name
CLOUDFRONT_DOMAIN=your-distribution.cloudfront.net
LAMBDA_FUNCTION_NAME=video-embedding-pipeline
```

## 2. MongoDB Atlas

Create a cluster and database, then create vector search indexes:

```bash
python scripts/create_mongodb_indexes.py
```

This creates three vector search indexes (one per modality collection):
- `visual_embeddings` → `visual_embeddings_vector_index`
- `audio_embeddings` → `audio_embeddings_vector_index`
- `transcription_embeddings` → `transcription_embeddings_vector_index`

Each index: 512 dimensions, cosine similarity, with `video_id` filter field.

**Network access:** Whitelist your App Runner VPC NAT gateway IP in Atlas (or use VPC peering for production).

## 3. S3 and CloudFront

Create an S3 bucket with this folder structure:
```
your-bucket/
├── input/          # Upload videos here (Lambda moves them after processing)
├── originals/      # Original source files (preserved by Lambda)
├── proxies/        # Web-friendly transcoded versions (MP4, H.264)
├── embeddings/     # Bedrock async embedding outputs
└── status/         # Upload progress markers
```

Create a CloudFront distribution pointing at the S3 bucket for video delivery.

## 4. Deploy Lambda

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh lambda
```

This packages `src/lambda_function.py`, `src/bedrock_client.py`, `src/mongodb_client.py`, and `src/compare_client.py` into a zip and deploys to Lambda.

Set these environment variables on the Lambda function (via console or CLI):
- `MONGODB_URI`
- `MONGODB_DATABASE`
- `S3_BUCKET`
- `CLOUDFRONT_DOMAIN`

Lambda requires IAM permissions for: S3, Bedrock, CloudWatch Logs.

## 5. Deploy App Runner

Connect your GitHub repo to App Runner (source-based deployment):

- **Repository:** `https://github.com/twelvelabs-io/multi-modal-video-search`
- **Branch:** `main`
- **Build config:** Uses `apprunner.yaml` in the repo
- **Port:** 8000

Set runtime environment variables via the App Runner console (not in the yaml):
- `MONGODB_URI`
- `MONGODB_DATABASE`
- `S3_BUCKET`
- `CLOUDFRONT_DOMAIN`
- `LAMBDA_FUNCTION_NAME`
- `AWS_REGION`

If App Runner needs to reach MongoDB Atlas through a NAT gateway (for IP whitelisting), configure a VPC connector on private subnets with a NAT gateway route.

## 6. Test

Upload a video:
```bash
aws s3 cp test-video.mp4 s3://${S3_BUCKET}/input/
```

Trigger Lambda:
```bash
aws lambda invoke \
  --function-name video-embedding-pipeline \
  --region us-east-1 \
  --payload '{"s3_key": "input/test-video.mp4", "bucket": "'${S3_BUCKET}'"}' \
  --cli-binary-format raw-in-base64-out \
  response.json
```

Open the web UI at your App Runner URL.

## Architecture

```
S3 Bucket (Videos)
    │
    ▼
Lambda (Processing)
    │  Bedrock Marengo 3.0
    │  512-dim embeddings
    │  visual + audio + transcription
    ▼
MongoDB Atlas (Multi-Collection)
    ├── visual_embeddings
    ├── audio_embeddings
    ├── transcription_embeddings
    └── video_fingerprints
    │
    ▼
App Runner (FastAPI)
    ├── Search: RRF, Weighted, Dynamic fusion
    ├── Compare: Cosine + L2 + Combined similarity
    ├── Clustering: Agglomerative + t-SNE visualization
    └── Frame extraction + VMAF analysis
    │
    ▼
CloudFront (Video Delivery) → Browser (Web UI)
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MONGODB_URI` | Yes | — | MongoDB Atlas connection string |
| `MONGODB_DATABASE` | No | `video_search` | Database name |
| `AWS_REGION` | No | `us-east-1` | AWS region (must be us-east-1 for Marengo) |
| `S3_BUCKET` | Yes | — | S3 bucket for video storage |
| `CLOUDFRONT_DOMAIN` | Yes | — | CloudFront distribution domain |
| `LAMBDA_FUNCTION_NAME` | No | `video-embedding-pipeline` | Lambda function name |

## Security

- Never commit credentials to git — use App Runner console or AWS Secrets Manager
- Restrict MongoDB Atlas network access to your NAT gateway IP
- Use IAM roles (not access keys) for Lambda and App Runner
- CloudFront and App Runner enforce HTTPS by default
