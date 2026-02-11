#!/bin/bash
#
# AWS Lambda Deployment Script for Multi-Vector Video Embedding Pipeline
#
# This script creates the IAM role, Lambda function, and necessary permissions
# for the video embedding pipeline in us-east-1.
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - Python 3.11+ installed
#   - zip command available
#
# Usage:
#   ./scripts/deploy.sh [--update]
#
# Environment Variables (required):
#   MONGODB_URI - MongoDB Atlas connection string
#
# Optional Environment Variables:
#   MONGODB_DATABASE - Database name (default: video_search)
#   LAMBDA_FUNCTION_NAME - Function name (default: video-embedding-pipeline)
#   AWS_REGION - AWS region (default: us-east-1)

set -e

# Configuration
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-video-embedding-pipeline}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ROLE_NAME="${LAMBDA_FUNCTION_NAME}-role"
POLICY_NAME="${LAMBDA_FUNCTION_NAME}-policy"
RUNTIME="python3.11"
HANDLER="lambda_function.lambda_handler"
TIMEOUT=900  # 15 minutes for video processing
MEMORY_SIZE=1024  # 1GB RAM

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check for required environment variable
if [ -z "$MONGODB_URI" ]; then
    echo_error "MONGODB_URI environment variable is required"
    echo "Export it with: export MONGODB_URI='mongodb+srv://...'"
    exit 1
fi

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/src"
BUILD_DIR="$PROJECT_DIR/build"

echo_info "Project directory: $PROJECT_DIR"
echo_info "Lambda function: $LAMBDA_FUNCTION_NAME"
echo_info "AWS Region: $AWS_REGION"

# Create build directory
mkdir -p "$BUILD_DIR"

# Check if we're updating or creating new
UPDATE_MODE=false
if [ "$1" == "--update" ]; then
    UPDATE_MODE=true
    echo_info "Running in UPDATE mode"
fi

# Step 1: Create the IAM trust policy document
echo_info "Creating IAM trust policy..."
cat > "$BUILD_DIR/trust-policy.json" << 'EOF'
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

# Step 2: Create the IAM permissions policy
echo_info "Creating IAM permissions policy..."
cat > "$BUILD_DIR/permissions-policy.json" << EOF
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
            "Sid": "S3FullAccess",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:GetObjectAcl",
                "s3:GetObjectVersion",
                "s3:PutObject",
                "s3:PutObjectAcl",
                "s3:CopyObject",
                "s3:DeleteObject",
                "s3:DeleteObjectVersion",
                "s3:ListBucket",
                "s3:ListBucketVersions",
                "s3:GetBucketLocation"
            ],
            "Resource": [
                "arn:aws:s3:::${S3_BUCKET}",
                "arn:aws:s3:::${S3_BUCKET}/*"
            ]
        },
        {
            "Sid": "BedrockInvoke",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:StartAsyncInvoke",
                "bedrock:GetAsyncInvoke"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/twelvelabs.marengo-embed-3-0-v1:0",
                "arn:aws:bedrock:${AWS_REGION}:${AWS_ACCOUNT_ID}:inference-profile/*",
                "arn:aws:bedrock:${AWS_REGION}:${AWS_ACCOUNT_ID}:async-invoke/*"
            ]
        },
        {
            "Sid": "S3VectorsAccess",
            "Effect": "Allow",
            "Action": [
                "s3vectors:*"
            ],
            "Resource": "*"
        }
    ]
}
EOF

# Step 3: Create or update IAM role
if ! $UPDATE_MODE; then
    echo_info "Creating IAM role: $ROLE_NAME"

    # Check if role exists
    if aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
        echo_warn "Role already exists, skipping creation"
    else
        aws iam create-role \
            --role-name "$ROLE_NAME" \
            --assume-role-policy-document "file://$BUILD_DIR/trust-policy.json" \
            --description "IAM role for video embedding Lambda function" \
            --region "$AWS_REGION"

        # Wait for role to be available
        echo_info "Waiting for role to be available..."
        sleep 10
    fi

    # Attach inline policy
    echo_info "Attaching permissions policy..."
    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "$POLICY_NAME" \
        --policy-document "file://$BUILD_DIR/permissions-policy.json"

    # Get role ARN
    ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
    echo_info "Role ARN: $ROLE_ARN"
else
    ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
    echo_info "Using existing role: $ROLE_ARN"
fi

# Step 4: Package Lambda function
echo_info "Packaging Lambda function..."

# Create package directory
PACKAGE_DIR="$BUILD_DIR/package"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

# Install dependencies
echo_info "Installing Python dependencies..."
pip install \
    --platform manylinux2014_x86_64 \
    --target "$PACKAGE_DIR" \
    --implementation cp \
    --python-version 3.11 \
    --only-binary=:all: \
    --upgrade \
    boto3 pymongo python-dotenv numpy \
    -q

# Copy source files
echo_info "Copying source files..."
cp "$SRC_DIR/lambda_function.py" "$PACKAGE_DIR/"
cp "$SRC_DIR/bedrock_client.py" "$PACKAGE_DIR/"
cp "$SRC_DIR/mongodb_client.py" "$PACKAGE_DIR/"
cp "$SRC_DIR/s3_vectors_client.py" "$PACKAGE_DIR/"

# Create ZIP file
echo_info "Creating deployment package..."
cd "$PACKAGE_DIR"
zip -r "$BUILD_DIR/lambda-package.zip" . -q
cd "$PROJECT_DIR"

PACKAGE_SIZE=$(du -h "$BUILD_DIR/lambda-package.zip" | cut -f1)
echo_info "Package size: $PACKAGE_SIZE"

# Step 5: Create or update Lambda function
if ! $UPDATE_MODE; then
    echo_info "Creating Lambda function: $LAMBDA_FUNCTION_NAME"

    # Check if function exists
    if aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$AWS_REGION" 2>/dev/null; then
        echo_warn "Function already exists, updating code..."
        aws lambda update-function-code \
            --function-name "$LAMBDA_FUNCTION_NAME" \
            --zip-file "fileb://$BUILD_DIR/lambda-package.zip" \
            --region "$AWS_REGION" \
            --output text > /dev/null
    else
        # Need to wait a bit more for role propagation
        echo_info "Waiting for IAM role propagation..."
        sleep 15

        aws lambda create-function \
            --function-name "$LAMBDA_FUNCTION_NAME" \
            --runtime "$RUNTIME" \
            --role "$ROLE_ARN" \
            --handler "$HANDLER" \
            --zip-file "fileb://$BUILD_DIR/lambda-package.zip" \
            --timeout "$TIMEOUT" \
            --memory-size "$MEMORY_SIZE" \
            --environment "Variables={MONGODB_URI=$MONGODB_URI,MONGODB_DATABASE=${MONGODB_DATABASE:-video_search},S3_BUCKET=${S3_BUCKET},S3_VECTORS_BUCKET=${S3_VECTORS_BUCKET:-},CLOUDFRONT_DOMAIN=${CLOUDFRONT_DOMAIN}}" \
            --region "$AWS_REGION" \
            --output text > /dev/null
    fi
else
    echo_info "Updating Lambda function code..."
    aws lambda update-function-code \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --zip-file "fileb://$BUILD_DIR/lambda-package.zip" \
        --region "$AWS_REGION" \
        --output text > /dev/null

    echo_info "Waiting for function update to complete..."
    aws lambda wait function-updated \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --region "$AWS_REGION"

    echo_info "Updating Lambda function configuration..."
    aws lambda update-function-configuration \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --timeout "$TIMEOUT" \
        --memory-size "$MEMORY_SIZE" \
        --environment "Variables={MONGODB_URI=$MONGODB_URI,MONGODB_DATABASE=${MONGODB_DATABASE:-video_search},S3_BUCKET=${S3_BUCKET},S3_VECTORS_BUCKET=${S3_VECTORS_BUCKET:-},CLOUDFRONT_DOMAIN=${CLOUDFRONT_DOMAIN}}" \
        --region "$AWS_REGION" \
        --output text > /dev/null
fi

# Step 6: Verify deployment
echo_info "Verifying deployment..."
FUNCTION_ARN=$(aws lambda get-function \
    --function-name "$LAMBDA_FUNCTION_NAME" \
    --region "$AWS_REGION" \
    --query 'Configuration.FunctionArn' \
    --output text)

echo ""
echo_info "=========================================="
echo_info "Deployment Complete!"
echo_info "=========================================="
echo ""
echo "Lambda Function: $LAMBDA_FUNCTION_NAME"
echo "Function ARN:    $FUNCTION_ARN"
echo "Region:          $AWS_REGION"
echo "Runtime:         $RUNTIME"
echo "Timeout:         ${TIMEOUT}s"
echo "Memory:          ${MEMORY_SIZE}MB"
echo ""
echo "Test the function with:"
echo ""
echo "  aws lambda invoke \\"
echo "    --function-name $LAMBDA_FUNCTION_NAME \\"
echo "    --region $AWS_REGION \\"
echo "    --payload '{\"s3_key\": \"input/test.mp4\", \"bucket\": \"your-media-bucket-name\"}' \\"
echo "    --cli-binary-format raw-in-base64-out \\"
echo "    response.json && cat response.json"
echo ""

# Cleanup build directory (optional)
# rm -rf "$BUILD_DIR"

echo_info "Done!"
