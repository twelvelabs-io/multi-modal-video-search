# Complete Setup Guide

This guide will help you set up the entire multi-modal video search system from scratch.

## Overview

The system consists of:
1. **AWS Lambda** - Processes videos and generates embeddings
2. **MongoDB Atlas** or **S3 Vectors** - Stores embeddings and metadata
3. **AWS S3** - Stores video files
4. **AWS CloudFront** - Delivers videos to users
5. **AWS App Runner** - Hosts the search API
6. **Web UI** - Frontend for searching videos

## Prerequisites

### Required
- AWS account with admin access
- AWS CLI installed and configured (`aws configure`)
- Python 3.11 or higher
- MongoDB Atlas account (free tier available)
- Git and GitHub account

### Optional
- Docker (for local development)
- Terraform or CloudFormation (for infrastructure as code)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/multi-modal-video-search.git
cd multi-modal-video-search
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your values
nano .env  # or vim, code, etc.
```

Required variables:
```bash
# MongoDB Atlas connection string
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/?appName=Cluster0

# AWS configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012

# S3 buckets
S3_BUCKET=your-media-bucket-name
S3_VECTORS_BUCKET=your-vectors-bucket-name  # Optional

# Lambda configuration
LAMBDA_FUNCTION_NAME=video-embedding-pipeline
LAMBDA_ROLE_NAME=video-embedding-pipeline-role
```

### 3. Run Setup Script

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run infrastructure setup
./scripts/setup_infrastructure.sh
```

This creates:
- ✅ S3 buckets for video storage
- ✅ IAM roles for Lambda
- ✅ CloudFront distribution
- ✅ Lambda function

### 4. Configure MongoDB

Follow [docs/mongodb-setup.md](docs/mongodb-setup.md) to:
- Create MongoDB Atlas cluster (M10+ tier)
- Set up network access
- Create database user
- Create vector search indexes

Or run the automated script:
```bash
python scripts/create_mongodb_indexes.py
```

### 5. Deploy App Runner

Follow [docs/apprunner-setup.md](docs/apprunner-setup.md) to deploy the search API.

Quick version via AWS Console:
1. Go to AWS App Runner
2. Connect to your GitHub repository
3. Configure build and environment variables
4. Deploy (takes ~3-5 minutes)

### 6. Test the System

Upload a test video:
```bash
# Upload video to S3
aws s3 cp test-video.mp4 s3://${S3_BUCKET}/WBD_project/Videos/Ready/

# Trigger Lambda processing
aws lambda invoke \
  --function-name ${LAMBDA_FUNCTION_NAME} \
  --region ${AWS_REGION} \
  --payload '{"s3_key": "WBD_project/Videos/Ready/test-video.mp4", "bucket": "'${S3_BUCKET}'"}' \
  --cli-binary-format raw-in-base64-out \
  response.json

# Check response
cat response.json
```

Access the web UI:
```
https://your-apprunner-service.us-east-1.awsapprunner.com
```

## Detailed Setup Guides

### MongoDB Setup
See [docs/mongodb-setup.md](docs/mongodb-setup.md) for:
- Cluster creation and configuration
- Vector index creation (4 indexes required)
- Network access setup
- Troubleshooting tips

### S3 Vectors Setup (Optional)
See README.md for:
- Creating S3 Vectors bucket
- Creating vector indexes (3 indexes)
- IAM permissions configuration

### App Runner Deployment
See [docs/apprunner-setup.md](docs/apprunner-setup.md) for:
- GitHub connection setup
- Service configuration
- Auto-deployment setup
- Monitoring and troubleshooting

## Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   S3 Bucket │────────▶│ Lambda       │────────▶│  MongoDB    │
│   (Videos)  │         │ (Processing) │         │  (Vectors)  │
└─────────────┘         └──────────────┘         └─────────────┘
                               │                         │
                               │                         │
                               ▼                         │
                        ┌──────────────┐                 │
                        │  CloudFront  │                 │
                        │  (Delivery)  │                 │
                        └──────────────┘                 │
                               │                         │
                               ▼                         ▼
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Browser   │────────▶│  App Runner  │────────▶│  Search API │
│   (Web UI)  │         │  (API Host)  │         │             │
└─────────────┘         └──────────────┘         └─────────────┘
```

## Data Flow

1. **Video Upload**: User uploads video to S3 bucket
2. **Processing**: Lambda function:
   - Downloads video from S3
   - Segments video using dynamic shot boundary detection
   - Generates embeddings with Bedrock Marengo 3.0
   - Stores embeddings in MongoDB/S3 Vectors
   - Moves video to proxy folder
3. **Search**: User enters search query in web UI:
   - API generates query embedding
   - Searches vector database (MongoDB or S3 Vectors)
   - Fuses results using selected method (RRF, Dynamic, Weighted, Fused)
   - Returns ranked video segments
4. **Playback**: User clicks result:
   - Web UI requests video from CloudFront
   - CloudFront serves video from S3
   - Video plays at specific timestamp

## Environment Variables Reference

### Required
- `MONGODB_URI` - MongoDB Atlas connection string
- `AWS_REGION` - AWS region (must be us-east-1 for Bedrock Marengo)
- `S3_BUCKET` - S3 bucket for video storage
- `CLOUDFRONT_DOMAIN` - CloudFront distribution domain

### Optional
- `MONGODB_DATABASE` - Database name (default: video_search)
- `S3_VECTORS_BUCKET` - S3 Vectors bucket name
- `LAMBDA_FUNCTION_NAME` - Lambda function name
- `WEIGHT_VISUAL` - Visual modality weight (default: 0.8)
- `WEIGHT_AUDIO` - Audio modality weight (default: 0.1)
- `WEIGHT_TRANSCRIPTION` - Transcription weight (default: 0.05)

## Cost Estimate

### Monthly Costs (Development)
- **MongoDB Atlas** (M10): $57/month
- **AWS Lambda** (1M invocations): ~$10/month
- **AWS S3** (100 GB storage): ~$2.30/month
- **AWS CloudFront** (100 GB transfer): ~$8.50/month
- **AWS App Runner** (1 vCPU, 2 GB): ~$40/month
- **Bedrock Embeddings** (10K queries): ~$100/month
- **Total**: ~$220/month

### Monthly Costs (Production)
- **MongoDB Atlas** (M30): $285/month
- **AWS Lambda** (10M invocations): ~$100/month
- **AWS S3** (1 TB storage): ~$23/month
- **AWS CloudFront** (1 TB transfer): ~$85/month
- **AWS App Runner** (2 vCPU, 4 GB, 3 instances): ~$240/month
- **Bedrock Embeddings** (100K queries): ~$1,000/month
- **Total**: ~$1,730/month

## Security Best Practices

1. **Never commit secrets to git**
   - Add `.env` to `.gitignore`
   - Use AWS Secrets Manager for production

2. **Use IAM roles instead of access keys**
   - Lambda uses execution role
   - App Runner uses service role

3. **Restrict MongoDB network access**
   - Whitelist specific IPs
   - Use VPC peering for production

4. **Enable CloudWatch logging**
   - Monitor Lambda executions
   - Track API errors

5. **Use HTTPS everywhere**
   - CloudFront enforces HTTPS
   - App Runner uses HTTPS by default

## Troubleshooting

### Lambda Errors
```bash
# View Lambda logs
aws logs tail /aws/lambda/${LAMBDA_FUNCTION_NAME} --follow

# Test Lambda function
aws lambda invoke \
  --function-name ${LAMBDA_FUNCTION_NAME} \
  --payload '{"test": "connection"}' \
  response.json
```

### MongoDB Connection Issues
```bash
# Test MongoDB connection
python scripts/verify_mongodb_setup.py
```

### App Runner Issues
```bash
# View App Runner logs
aws logs tail /aws/apprunner/${APP_RUNNER_SERVICE_NAME}/service --follow

# Trigger manual deployment
aws apprunner start-deployment --service-arn ${SERVICE_ARN}
```

## Next Steps

1. **Process Videos**: Upload videos to S3 and run Lambda
2. **Test Search**: Try different search queries and fusion methods
3. **Optimize Performance**: Tune weights, adjust cluster size
4. **Monitor Costs**: Set up billing alerts
5. **Scale Up**: Add more App Runner instances, upgrade MongoDB tier

## Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/YOUR_REPO/issues)
- **Documentation**: See `docs/` folder for detailed guides
- **Examples**: See `examples/` folder for code samples

## License

[Your License Here]
