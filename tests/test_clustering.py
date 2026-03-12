"""Tests for video clustering module."""
import numpy as np
import pytest
from src.clustering import cluster_videos, compute_2d_positions


def _make_video_embedding(base, noise_scale=0.05):
    """Create a 512d embedding near a base vector with small noise."""
    rng = np.random.RandomState(42)
    return (np.array(base) + rng.randn(512) * noise_scale).tolist()


# Two clusters: 3 similar videos + 1 outlier
# Use explicitly orthogonal vectors to guarantee separation
BASE_A = np.zeros(512); BASE_A[0] = 1.0; BASE_A = BASE_A.tolist()
BASE_B = np.zeros(512); BASE_B[256] = 1.0; BASE_B = BASE_B.tolist()  # orthogonal to A

VIDEOS = {
    "vid_a1": _make_video_embedding(BASE_A, 0.02),
    "vid_a2": _make_video_embedding(BASE_A, 0.02),
    "vid_a3": _make_video_embedding(BASE_A, 0.02),
    "vid_b1": _make_video_embedding(BASE_B, 0.02),
}


class TestClusterVideos:
    def test_returns_list_of_cluster_dicts(self):
        clusters = cluster_videos(VIDEOS)
        assert isinstance(clusters, list)
        assert len(clusters) >= 1
        for c in clusters:
            assert "id" in c
            assert "video_ids" in c
            assert "centroid" in c
            assert "avg_similarity" in c
            assert isinstance(c["video_ids"], list)
            assert len(c["centroid"]) == 512

    def test_groups_similar_videos_together(self):
        clusters = cluster_videos(VIDEOS)
        # Find the cluster containing vid_a1
        a_cluster = next(c for c in clusters if "vid_a1" in c["video_ids"])
        # All three A videos should be in the same cluster
        assert "vid_a2" in a_cluster["video_ids"]
        assert "vid_a3" in a_cluster["video_ids"]
        # B should NOT be in the A cluster
        assert "vid_b1" not in a_cluster["video_ids"]

    def test_singleton_cluster_for_outlier(self):
        clusters = cluster_videos(VIDEOS)
        b_cluster = next(c for c in clusters if "vid_b1" in c["video_ids"])
        assert len(b_cluster["video_ids"]) == 1

    def test_avg_similarity_range(self):
        clusters = cluster_videos(VIDEOS)
        for c in clusters:
            if len(c["video_ids"]) > 1:
                assert 0.0 <= c["avg_similarity"] <= 1.0
            else:
                # Singletons have no pairwise similarity
                assert c["avg_similarity"] == 1.0

    def test_empty_input(self):
        clusters = cluster_videos({})
        assert clusters == []

    def test_single_video(self):
        clusters = cluster_videos({"only_vid": BASE_A})
        assert len(clusters) == 1
        assert clusters[0]["video_ids"] == ["only_vid"]

    def test_custom_distance_threshold(self):
        # Very tight threshold — should split into more clusters
        clusters_tight = cluster_videos(VIDEOS, distance_threshold=0.01)
        clusters_loose = cluster_videos(VIDEOS, distance_threshold=0.9)
        assert len(clusters_tight) >= len(clusters_loose)


class TestCompute2dPositions:
    def test_returns_positions_for_each_cluster(self):
        clusters = cluster_videos(VIDEOS)
        positions = compute_2d_positions(clusters)
        assert len(positions) == len(clusters)
        for cid, pos in positions.items():
            assert "x" in pos and "y" in pos
            assert 0.0 <= pos["x"] <= 1.0
            assert 0.0 <= pos["y"] <= 1.0

    def test_single_cluster_centered(self):
        clusters = cluster_videos({"vid": BASE_A})
        positions = compute_2d_positions(clusters)
        # Single cluster should be at center (0.5, 0.5)
        pos = list(positions.values())[0]
        assert pos["x"] == 0.5
        assert pos["y"] == 0.5

    def test_two_clusters_separated(self):
        clusters = cluster_videos(VIDEOS)
        if len(clusters) >= 2:
            positions = compute_2d_positions(clusters)
            coords = list(positions.values())
            # Two clusters shouldn't be at exact same position
            dist = ((coords[0]["x"] - coords[1]["x"])**2 + (coords[0]["y"] - coords[1]["y"])**2)**0.5
            assert dist > 0.05  # Meaningfully separated
