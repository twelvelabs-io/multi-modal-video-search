"""
Video clustering module.

Groups videos by embedding similarity using agglomerative clustering
with cosine distance. Produces cluster objects with centroids,
avg similarity, and 2D positions for visualization.
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normalized = vectors / norms
    return normalized @ normalized.T


def cluster_videos(
    video_embeddings: dict[str, list],
    distance_threshold: float = 0.3
) -> list[dict]:
    """
    Cluster videos by embedding similarity.

    Args:
        video_embeddings: {video_id: 512d_embedding_list}
        distance_threshold: Cosine distance threshold for grouping (0.3 = 0.7 similarity)

    Returns:
        List of cluster dicts with keys:
        - id: "cluster_0", "cluster_1", ...
        - video_ids: list of video_id strings
        - centroid: 512d list (mean of member embeddings)
        - avg_similarity: mean pairwise cosine similarity within cluster (1.0 for singletons)
    """
    if not video_embeddings:
        return []

    video_ids = list(video_embeddings.keys())
    vectors = np.array([video_embeddings[vid] for vid in video_ids], dtype=np.float64)

    if len(video_ids) == 1:
        return [{
            "id": "cluster_0",
            "video_ids": video_ids,
            "centroid": vectors[0].tolist(),
            "avg_similarity": 1.0
        }]

    # Agglomerative clustering with cosine distance
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(vectors)

    # Build clusters
    sim_matrix = cosine_similarity_matrix(vectors)
    clusters = []

    for label in sorted(set(labels)):
        member_mask = labels == label
        member_indices = np.where(member_mask)[0]
        member_ids = [video_ids[i] for i in member_indices]
        member_vectors = vectors[member_indices]

        centroid = member_vectors.mean(axis=0)

        # Average pairwise similarity within cluster
        if len(member_indices) > 1:
            sub_sim = sim_matrix[np.ix_(member_indices, member_indices)]
            # Upper triangle (excluding diagonal)
            triu = np.triu_indices(len(member_indices), k=1)
            avg_sim = float(sub_sim[triu].mean())
        else:
            avg_sim = 1.0

        clusters.append({
            "id": f"cluster_{label}",
            "video_ids": member_ids,
            "centroid": centroid.tolist(),
            "avg_similarity": round(avg_sim, 4)
        })

    return clusters


def compute_2d_positions(clusters: list[dict]) -> dict[str, dict]:
    """
    Project cluster centroids to 2D for visualization.

    Args:
        clusters: Output from cluster_videos()

    Returns:
        {cluster_id: {"x": 0.0-1.0, "y": 0.0-1.0}}
    """
    if not clusters:
        return {}

    if len(clusters) == 1:
        return {clusters[0]["id"]: {"x": 0.5, "y": 0.5}}

    centroids = np.array([c["centroid"] for c in clusters], dtype=np.float64)

    if len(clusters) == 2:
        # t-SNE needs perplexity < n_samples; for 2 points just spread them
        return {
            clusters[0]["id"]: {"x": 0.3, "y": 0.5},
            clusters[1]["id"]: {"x": 0.7, "y": 0.5},
        }

    # t-SNE projection
    perplexity = min(5, len(clusters) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        metric="cosine",
        init="random"
    )
    coords_2d = tsne.fit_transform(centroids)

    # Normalize to 0-1 with padding
    padding = 0.1
    for dim in range(2):
        col = coords_2d[:, dim]
        mn, mx = col.min(), col.max()
        span = mx - mn if mx != mn else 1.0
        coords_2d[:, dim] = padding + (col - mn) / span * (1 - 2 * padding)

    positions = {}
    for i, c in enumerate(clusters):
        positions[c["id"]] = {
            "x": round(float(coords_2d[i, 0]), 4),
            "y": round(float(coords_2d[i, 1]), 4)
        }

    return positions
