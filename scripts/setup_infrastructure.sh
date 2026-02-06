#!/bin/bash
#
# Infrastructure Setup Script for Multi-Modal Video Search
#
# This script creates all required AWS infrastructure:
# - S3 buckets (media storage + S3 Vectors)
# - IAM roles (Lambda execution, MediaConvert)
# - CloudFront distribution
# - Lambda function
# - App Runner service (optional)
#
# Prerequisites:
# - AWS CLI configured with admin access
# - .env file with configuration (copy from .env.example)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo_step() {
    echo -e "${BLUE}==>${NC} $1"
}

# Check for .env file
if [ ! -f .env ]; then
    echo_error ".env file not found!"
    echo "Please copy .env.example to .env and configure it:"
    echo "  cp .env.example .env"
    echo "  # Edit .env with your values"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Validate required variables
REQUIRED_VARS=(
    "AWS_REGION"
    "AWS_ACCOUNT_ID"
    "S3_BUCKET"
    "MONGODB_URI"
    "LAMBDA_FUNCTION_NAME"
    "LAMBDA_ROLE_NAME"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo_error "Required variable $var is not set in .env"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "  AWS Infrastructure Setup"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  AWS Region:        ${AWS_REGION}"
echo "  AWS Account:       ${AWS_ACCOUNT_ID}"
echo "  S3 Bucket:         ${S3_BUCKET}"
echo "  Lambda Function:   ${LAMBDA_FUNCTION_NAME}"
echo "  Lambda Role:       ${LAMBDA_ROLE_NAME}"
echo ""
read -p "Continue with setup? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi
echo ""

# Step 1: Create S3 Bucket for Media
echo_step "Step 1: Creating S3 bucket for media storage"

if aws s3 ls "s3://${S3_BUCKET}" 2>/dev/null; then
    echo_warn "S3 bucket ${S3_BUCKET} already exists, skipping..."
else
    aws s3 mb "s3://${S3_BUCKET}" --region ${AWS_REGION}
    echo_info "Created S3 bucket: ${S3_BUCKET}"

    # Enable versioning
    aws s3api put-bucket-versioning \
        --bucket ${S3_BUCKET} \
        --versioning-configuration Status=Enabled

    # Create folder structure
    aws s3api put-object --bucket ${S3_BUCKET} --key WBD_project/Videos/
    aws s3api put-object --bucket ${S3_BUCKET} --key WBD_project/Videos/proxy/
    aws s3api put-object --bucket ${S3_BUCKET} --key WBD_project/Videos/Ready/

    echo_info "Created folder structure"
fi

# Step 2: Create S3 Vectors bucket (if configured)
if [ -n "${S3_VECTORS_BUCKET}" ] && [ "${S3_VECTORS_BUCKET}" != "your-vectors-bucket-name" ]; then
    echo_step "Step 2: Creating S3 Vectors bucket"

    if aws s3 ls "s3://${S3_VECTORS_BUCKET}" 2>/dev/null; then
        echo_warn "S3 Vectors bucket ${S3_VECTORS_BUCKET} already exists, skipping..."
    else
        aws s3 mb "s3://${S3_VECTORS_BUCKET}" --region ${AWS_REGION}
        echo_info "Created S3 Vectors bucket: ${S3_VECTORS_BUCKET}"
    fi
else
    echo_step "Step 2: Skipping S3 Vectors bucket (not configured)"
fi

# Step 3: Create IAM role for Lambda
echo_step "Step 3: Creating IAM role for Lambda"

if aws iam get-role --role-name ${LAMBDA_ROLE_NAME} 2>/dev/null; then
    echo_warn "IAM role ${LAMBDA_ROLE_NAME} already exists, skipping..."
    ROLE_ARN=$(aws iam get-role --role-name ${LAMBDA_ROLE_NAME} --query 'Role.Arn' --output text)
else
    # Create trust policy
    cat > /tmp/lambda-trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

    # Create role
    aws iam create-role \
        --role-name ${LAMBDA_ROLE_NAME} \
        --assume-role-policy-document file:///tmp/lambda-trust-policy.json \
        --description "IAM role for video embedding Lambda function"

    ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"

    # Create permissions policy
    cat > /tmp/lambda-permissions-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:${AWS_REGION}:*:*"
        },
        {
            "Sid": "S3Access",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${S3_BUCKET}/*",
                "arn:aws:s3:::${S3_BUCKET}"
            ]
        },
        {
            "Sid": "BedrockInvoke",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": [
                "arn:aws:bedrock:${AWS_REGION}::foundation-model/twelvelabs.marengo-embed-3-0-v1:0",
                "arn:aws:bedrock:${AWS_REGION}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
            ]
        }
    ]
}
EOF

    # Attach permissions policy
    aws iam put-role-policy \
        --role-name ${LAMBDA_ROLE_NAME} \
        --policy-name "${LAMBDA_ROLE_NAME}-policy" \
        --policy-document file:///tmp/lambda-permissions-policy.json

    echo_info "Created IAM role: ${ROLE_ARN}"

    # Wait for role propagation
    echo_info "Waiting for IAM role propagation (10 seconds)..."
    sleep 10
fi

# Step 4: Create CloudFront distribution (optional)
if [ -z "${CLOUDFRONT_DOMAIN}" ] || [ "${CLOUDFRONT_DOMAIN}" == "xxxxx.cloudfront.net" ]; then
    echo_step "Step 4: Creating CloudFront distribution"

    # Create CloudFront distribution
    cat > /tmp/cloudfront-config.json << EOF
{
    "CallerReference": "video-search-$(date +%s)",
    "Comment": "CloudFront distribution for video search",
    "Enabled": true,
    "Origins": {
        "Quantity": 1,
        "Items": [
            {
                "Id": "S3-${S3_BUCKET}",
                "DomainName": "${S3_BUCKET}.s3.amazonaws.com",
                "S3OriginConfig": {
                    "OriginAccessIdentity": ""
                }
            }
        ]
    },
    "DefaultCacheBehavior": {
        "TargetOriginId": "S3-${S3_BUCKET}",
        "ViewerProtocolPolicy": "redirect-to-https",
        "TrustedSigners": {
            "Enabled": false,
            "Quantity": 0
        },
        "ForwardedValues": {
            "QueryString": false,
            "Cookies": {
                "Forward": "none"
            }
        },
        "MinTTL": 0,
        "DefaultTTL": 86400,
        "MaxTTL": 31536000
    }
}
EOF

    DISTRIBUTION_ID=$(aws cloudfront create-distribution \
        --distribution-config file:///tmp/cloudfront-config.json \
        --query 'Distribution.Id' \
        --output text)

    CLOUDFRONT_DOMAIN=$(aws cloudfront get-distribution \
        --id ${DISTRIBUTION_ID} \
        --query 'Distribution.DomainName' \
        --output text)

    echo_info "Created CloudFront distribution: ${CLOUDFRONT_DOMAIN}"
    echo_warn "Update your .env file with: CLOUDFRONT_DOMAIN=${CLOUDFRONT_DOMAIN}"
else
    echo_step "Step 4: Using existing CloudFront distribution: ${CLOUDFRONT_DOMAIN}"
fi

# Step 5: Deploy Lambda function
echo_step "Step 5: Deploying Lambda function"

echo_info "Running Lambda deployment script..."
export MONGODB_URI
export MONGODB_DATABASE
export LAMBDA_FUNCTION_NAME
export AWS_REGION
export S3_VECTORS_BUCKET

./scripts/deploy.sh

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. MongoDB Setup:"
echo "   - Create MongoDB Atlas cluster (M10+ tier)"
echo "   - Update .env with MONGODB_URI"
echo "   - Create vector indexes (see docs/mongodb-setup.md)"
echo ""
echo "2. S3 Vectors Setup (optional):"
echo "   - Create vector indexes:"
echo "     aws s3vectors create-index --index-name visual-embeddings ..."
echo "   - See docs/s3vectors-setup.md for details"
echo ""
echo "3. Test Lambda:"
echo "   aws lambda invoke \\"
echo "     --function-name ${LAMBDA_FUNCTION_NAME} \\"
echo "     --region ${AWS_REGION} \\"
echo "     --payload '{\"test\": \"connection\"}' \\"
echo "     response.json"
echo ""
echo "4. Deploy App Runner:"
echo "   - See docs/apprunner-setup.md"
echo ""
echo "CloudFront Domain: ${CLOUDFRONT_DOMAIN}"
echo "Lambda Function:   ${LAMBDA_FUNCTION_NAME}"
echo "S3 Bucket:         ${S3_BUCKET}"
echo ""
