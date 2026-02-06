#!/usr/bin/env python3
"""
Create MongoDB Atlas Vector Search Indexes

This script creates all required vector search indexes for the video search system.
Requires pymongo and MongoDB Atlas M10+ cluster.
"""
import os
import sys
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel


def create_vector_indexes():
    """Create all vector search indexes."""

    # Load configuration from environment
    mongodb_uri = os.environ.get("MONGODB_URI")
    mongodb_database = os.environ.get("MONGODB_DATABASE", "video_search")

    if not mongodb_uri:
        print("❌ MONGODB_URI environment variable not set")
        print("Set it with: export MONGODB_URI='mongodb+srv://...'")
        return False

    print("=" * 80)
    print("CREATING MONGODB ATLAS VECTOR SEARCH INDEXES")
    print("=" * 80)
    print()
    print(f"Database: {mongodb_database}")
    print()

    try:
        client = MongoClient(mongodb_uri)
        db = client[mongodb_database]

        # Test connection
        client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas")
        print()
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return False

    # Index definitions
    index_definitions = {
        'unified-embeddings': {
            'name': 'unified_embeddings_vector_index',
            'definition': {
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
        },
        'visual_embeddings': {
            'name': 'visual_embeddings_vector_index',
            'definition': {
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
        },
        'audio_embeddings': {
            'name': 'audio_embeddings_vector_index',
            'definition': {
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
        },
        'transcription_embeddings': {
            'name': 'transcription_embeddings_vector_index',
            'definition': {
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
        }
    }

    results = []

    for collection_name, index_info in index_definitions.items():
        print(f"Creating index for {collection_name}...")

        try:
            collection = db[collection_name]

            # Create search index using pymongo
            search_index = SearchIndexModel(
                definition=index_info['definition'],
                name=index_info['name'],
                type="vectorSearch"
            )

            result = collection.create_search_index(model=search_index)

            print(f"  ✅ Index '{index_info['name']}' created successfully")
            print(f"     Index ID: {result}")
            print(f"     Status: PENDING (building...)")
            print()

            results.append({
                'collection': collection_name,
                'index_name': index_info['name'],
                'status': 'created',
                'index_id': result
            })

        except Exception as e:
            error_msg = str(e)
            if 'Duplicate Index' in error_msg or 'already exists' in error_msg:
                print(f"  ⚠️  Index already exists")
                results.append({
                    'collection': collection_name,
                    'index_name': index_info['name'],
                    'status': 'already_exists'
                })
            else:
                print(f"  ❌ Error: {error_msg}")
                results.append({
                    'collection': collection_name,
                    'index_name': index_info['name'],
                    'status': 'error',
                    'error': error_msg
                })
            print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    created = sum(1 for r in results if r['status'] == 'created')
    existing = sum(1 for r in results if r['status'] == 'already_exists')
    failed = sum(1 for r in results if r['status'] == 'error')

    print(f"✅ Created: {created}")
    print(f"⚠️  Already existed: {existing}")
    print(f"❌ Failed: {failed}")
    print()

    if created > 0:
        print("⏱️  Indexes are now building (status: PENDING)")
        print("   This takes ~5-10 minutes per index")
        print("   Check status in Atlas UI: Cluster → Search tab")
        print()

    if failed > 0:
        print("❌ Some indexes failed to create. Check errors above.")
        print()

    client.close()
    return failed == 0


if __name__ == "__main__":
    # Load .env file if it exists
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()

    success = create_vector_indexes()
    sys.exit(0 if success else 1)
