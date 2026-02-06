#!/bin/bash
#
# Create S3 Vectors Indexes
#
# This script creates all required S3 Vectors indexes for the video search system.
# Requires AWS CLI and appropriate IAM permissions.
#

set -e

# Color output helpers
echo_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

echo_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

echo_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

echo_warn() {
    echo -e "\033[1;33m[WARN]\033[0m $1"
}

# Load environment variables
if [ -f .env ]; then
    echo_info "Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
fi

# Configuration
S3_VECTORS_BUCKET="${S3_VECTORS_BUCKET}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Validate required variables
if [ -z "$S3_VECTORS_BUCKET" ]; then
    echo_error "S3_VECTORS_BUCKET environment variable not set"
    echo "Set it with: export S3_VECTORS_BUCKET='your-bucket-name'"
    exit 1
fi

echo "================================================================================"
echo "CREATING S3 VECTORS INDEXES"
echo "================================================================================"
echo ""
echo "Bucket: $S3_VECTORS_BUCKET"
echo "Region: $AWS_REGION"
echo ""

# Check if bucket exists
echo_info "Checking if bucket exists..."
if ! aws s3 ls "s3://${S3_VECTORS_BUCKET}" --region "$AWS_REGION" 2>/dev/null; then
    echo_error "Bucket ${S3_VECTORS_BUCKET} does not exist"
    echo "Create it with: aws s3 mb s3://${S3_VECTORS_BUCKET} --region ${AWS_REGION}"
    exit 1
fi
echo_success "Bucket exists"
echo ""

# Index definitions
declare -A indexes=(
    ["visual-embeddings"]="Visual embeddings (scenes, actions, objects)"
    ["audio-embeddings"]="Audio embeddings (sounds, music, ambient)"
    ["transcription-embeddings"]="Transcription embeddings (spoken words, dialogue)"
)

EMBEDDING_DIMENSION=512
DISTANCE_METRIC="COSINE"

created=0
existing=0
failed=0

# Create each index
for index_name in "${!indexes[@]}"; do
    description="${indexes[$index_name]}"

    echo_info "Creating index: $index_name"
    echo "  Description: $description"
    echo "  Dimensions: $EMBEDDING_DIMENSION"
    echo "  Distance: $DISTANCE_METRIC"

    # Create the index
    result=$(aws s3-vectors create-vector-index \
        --bucket-name "$S3_VECTORS_BUCKET" \
        --index-name "$index_name" \
        --embedding-dimension "$EMBEDDING_DIMENSION" \
        --distance-metric "$DISTANCE_METRIC" \
        --region "$AWS_REGION" 2>&1) || true

    if echo "$result" | grep -q "already exists"; then
        echo_warn "Index already exists"
        ((existing++))
    elif echo "$result" | grep -q "error\|Error\|failed"; then
        echo_error "Failed to create index"
        echo "$result"
        ((failed++))
    else
        echo_success "Index created successfully"
        ((created++))
    fi
    echo ""
done

# Summary
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""
echo "✅ Created: $created"
echo "⚠️  Already existed: $existing"
echo "❌ Failed: $failed"
echo ""

if [ $failed -gt 0 ]; then
    echo_error "Some indexes failed to create. Check errors above."
    exit 1
fi

if [ $created -gt 0 ]; then
    echo_success "All indexes created successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Verify indexes: aws s3-vectors list-vector-indexes --bucket-name $S3_VECTORS_BUCKET --region $AWS_REGION"
    echo "  2. Update .env with S3_VECTORS_BUCKET=$S3_VECTORS_BUCKET"
    echo "  3. Deploy Lambda function: ./scripts/deploy.sh"
    echo "  4. Process videos to populate indexes"
fi

if [ $existing -eq ${#indexes[@]} ]; then
    echo_info "All indexes already exist. No action needed."
fi
