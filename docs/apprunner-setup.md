# AWS App Runner Setup Guide

This guide walks you through deploying the video search API to AWS App Runner.

## Prerequisites

- GitHub repository with the code
- AWS account with App Runner access
- MongoDB Atlas configured (see mongodb-setup.md)

## Option A: Using AWS Console (Recommended for First Time)

### Step 1: Connect GitHub

1. Go to [AWS App Runner Console](https://console.aws.amazon.com/apprunner/)
2. Click "Create service"
3. Select "Source code repository"
4. Click "Add new" to connect GitHub
5. Authorize AWS Connector for GitHub
6. Select your repository
7. Select branch: `main` (or your default branch)

### Step 2: Configure Build

1. Runtime: **Python 3**
2. Build command:
   ```bash
   pip3 install -r requirements.txt -t ./deps
   ```
3. Start command:
   ```bash
   python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
   ```
4. Port: **8000**

### Step 3: Configure Service

1. Service name: `video-search` (or your choice)
2. Virtual CPU: **1 vCPU**
3. Memory: **2 GB**
4. Auto scaling:
   - Min instances: 1
   - Max instances: 3 (adjust based on load)
   - Concurrency: 100

### Step 4: Configure Environment Variables

Add these environment variables:

```bash
AWS_REGION=us-east-1
CLOUDFRONT_DOMAIN=xxxxx.cloudfront.net
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/?appName=Cluster0
MONGODB_DATABASE=video_search
S3_VECTORS_BUCKET=your-vectors-bucket (optional)
PYTHONPATH=/app/deps
```

⚠️ **Security:** Store MONGODB_URI securely. Consider using AWS Secrets Manager.

### Step 5: Configure Auto-Deployment

1. Enable auto-deployments: **Yes**
2. Deployment trigger: **Automatic** (on git push to main branch)

### Step 6: Review and Create

1. Review all settings
2. Click "Create & deploy"
3. Wait for deployment (~3-5 minutes)

### Step 7: Get Service URL

After deployment completes, your service URL will be:
```
https://xxxxx.us-east-1.awsapprunner.com
```

Test it:
```bash
curl https://xxxxx.us-east-1.awsapprunner.com/api/health
```

## Option B: Using AWS CLI

### Step 1: Create GitHub Connection

First, create a GitHub connection (one-time setup):

1. Go to AWS Console → Developer Tools → Connections
2. Create connection to GitHub
3. Note the connection ARN

### Step 2: Create App Runner Service

```bash
# Load environment variables
source .env

# Create service configuration
cat > /tmp/apprunner-config.json << EOF
{
  "ServiceName": "${APP_RUNNER_SERVICE_NAME}",
  "SourceConfiguration": {
    "AuthenticationConfiguration": {
      "ConnectionArn": "YOUR_GITHUB_CONNECTION_ARN"
    },
    "AutoDeploymentsEnabled": true,
    "CodeRepository": {
      "RepositoryUrl": "${GITHUB_REPO_URL}",
      "SourceCodeVersion": {
        "Type": "BRANCH",
        "Value": "main"
      },
      "CodeConfiguration": {
        "ConfigurationSource": "API",
        "CodeConfigurationValues": {
          "Runtime": "PYTHON_3",
          "BuildCommand": "pip3 install -r requirements.txt -t ./deps",
          "StartCommand": "python3 -m uvicorn app:app --host 0.0.0.0 --port 8000",
          "Port": "8000",
          "RuntimeEnvironmentVariables": {
            "AWS_REGION": "${AWS_REGION}",
            "CLOUDFRONT_DOMAIN": "${CLOUDFRONT_DOMAIN}",
            "MONGODB_URI": "${MONGODB_URI}",
            "MONGODB_DATABASE": "${MONGODB_DATABASE}",
            "PYTHONPATH": "/app/deps"
          }
        }
      }
    }
  },
  "InstanceConfiguration": {
    "Cpu": "1 vCPU",
    "Memory": "2 GB"
  },
  "AutoScalingConfigurationArn": "arn:aws:apprunner:${AWS_REGION}:${AWS_ACCOUNT_ID}:autoscalingconfiguration/DefaultConfiguration/1/00000000000000000000000000000001"
}
EOF

# Create service
aws apprunner create-service \
  --cli-input-json file:///tmp/apprunner-config.json \
  --region ${AWS_REGION}
```

### Step 3: Wait for Deployment

```bash
# Get service ARN
SERVICE_ARN=$(aws apprunner list-services \
  --query "ServiceSummaryList[?ServiceName=='${APP_RUNNER_SERVICE_NAME}'].ServiceArn" \
  --output text)

# Wait for service to be running
aws apprunner wait service-running \
  --service-arn ${SERVICE_ARN}

# Get service URL
SERVICE_URL=$(aws apprunner describe-service \
  --service-arn ${SERVICE_ARN} \
  --query 'Service.ServiceUrl' \
  --output text)

echo "Service is running at: https://${SERVICE_URL}"
```

## Updating the Service

### Update Environment Variables

```bash
# Update via AWS Console
# App Runner → video-search → Configuration → Edit → Environment variables

# Or via CLI
aws apprunner update-service \
  --service-arn ${SERVICE_ARN} \
  --source-configuration '{ ... }' \
  --region ${AWS_REGION}
```

### Trigger Manual Deployment

```bash
aws apprunner start-deployment \
  --service-arn ${SERVICE_ARN} \
  --region ${AWS_REGION}
```

## Monitoring

### View Logs

```bash
# Via AWS Console
# App Runner → video-search → Logs

# Or stream via CLI
aws logs tail /aws/apprunner/${APP_RUNNER_SERVICE_NAME}/${SERVICE_ID}/service \
  --follow \
  --region ${AWS_REGION}
```

### Monitor Metrics

View in AWS Console → App Runner → video-search → Metrics:
- Request count
- Response time (p50, p90, p99)
- HTTP errors (4xx, 5xx)
- Active instances
- CPU/Memory utilization

## Troubleshooting

### Deployment Failed

Check deployment logs:
```bash
aws apprunner describe-service \
  --service-arn ${SERVICE_ARN} \
  --query 'Service.Status' \
  --region ${AWS_REGION}
```

Common issues:
- Missing environment variables
- Invalid MONGODB_URI
- Requirements.txt missing dependencies
- Build command failed

### Service Unhealthy

Check application logs for errors:
- MongoDB connection failures
- Missing AWS permissions
- Invalid configuration

### Slow Performance

- Increase instance size (1 vCPU → 2 vCPU)
- Increase memory (2 GB → 4 GB)
- Enable more concurrent requests
- Add more auto-scaling instances

## Cost Optimization

**App Runner Costs:**
- 1 vCPU, 2 GB: ~$40/month (1 instance, always on)
- Includes 100 GB network egress
- Additional vCPU: ~$6/month
- Additional GB memory: ~$3/month

**Recommendations:**
- Use 1 instance for development
- 2-3 instances for production
- Enable auto-scaling for traffic spikes
- Use pause/resume for non-production environments

## Security Best Practices

1. **Use Secrets Manager** for sensitive variables:
   ```bash
   # Store MONGODB_URI in Secrets Manager
   aws secretsmanager create-secret \
     --name video-search/mongodb-uri \
     --secret-string "${MONGODB_URI}"

   # Reference in App Runner (not yet supported, use env vars for now)
   ```

2. **Enable VPC connector** for private MongoDB access

3. **Use custom domain** with SSL certificate

4. **Enable WAF** for DDoS protection

## Next Steps

1. Test the API: `curl https://your-service.awsapprunner.com/api/health`
2. Upload a test video to S3
3. Process video with Lambda
4. Search in the web UI
5. Monitor performance and scale as needed
