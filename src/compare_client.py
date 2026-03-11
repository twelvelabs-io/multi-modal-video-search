"""Video comparison client for QC/duplicate detection."""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def compute_fingerprint(segments: list) -> dict:
    """Compute per-modality mean embeddings from segment list.

    Args:
        segments: List of segment dicts with 'modality_type' and 'embedding' keys.

    Returns:
        Dict with visual_fingerprint, audio_fingerprint, transcription_fingerprint,
        segment_count, total_duration.
    """
    by_modality = {"visual": [], "audio": [], "transcription": []}
    times = []
    segment_ids = set()

    for seg in segments:
        modality = seg.get("modality_type")
        embedding = seg.get("embedding")
        if modality in by_modality and embedding:
            by_modality[modality].append(embedding)
        seg_id = seg.get("segment_id")
        if seg_id is not None:
            segment_ids.add(seg_id)
        start = seg.get("start_time", 0)
        end = seg.get("end_time", 0)
        if end:
            times.append(end)

    result = {}
    for modality, embeddings in by_modality.items():
        if embeddings:
            mean = np.mean(embeddings, axis=0).tolist()
        else:
            mean = [0.0] * 512
        result[f"{modality}_fingerprint"] = mean

    result["segment_count"] = len(segment_ids)
    result["total_duration"] = max(times) if times else 0.0
    return result


def find_similar_videos(reference_id: str, all_fingerprints: list,
                        top_k: int = 20) -> list:
    """Find videos most similar to reference using fingerprint cosine similarity.

    Overall similarity = visual*0.5 + audio*0.25 + transcription*0.25

    Args:
        reference_id: video_id of the reference video.
        all_fingerprints: List of all fingerprint dicts from MongoDB.
        top_k: Maximum results to return.

    Returns:
        Sorted list of dicts with video metadata and similarity scores.
    """
    ref = None
    others = []
    for fp in all_fingerprints:
        if fp["video_id"] == reference_id:
            ref = fp
        else:
            others.append(fp)

    if ref is None:
        logger.error(f"Reference video {reference_id} not found in fingerprints")
        return []

    results = []
    for other in others:
        vis_sim = cosine_similarity(ref["visual_fingerprint"], other["visual_fingerprint"])
        aud_sim = cosine_similarity(ref["audio_fingerprint"], other["audio_fingerprint"])
        tra_sim = cosine_similarity(ref["transcription_fingerprint"], other["transcription_fingerprint"])
        overall = vis_sim * 0.5 + aud_sim * 0.25 + tra_sim * 0.25

        results.append({
            "video_id": other["video_id"],
            "name": other.get("video_name", other["video_id"]),
            "segment_count": other.get("segment_count", 0),
            "duration": other.get("total_duration", 0.0),
            "overall_similarity": round(overall, 4),
            "modality_scores": {
                "visual": round(vis_sim, 4),
                "audio": round(aud_sim, 4),
                "transcription": round(tra_sim, 4),
            }
        })

    results.sort(key=lambda x: x["overall_similarity"], reverse=True)
    return results[:top_k]


def align_segments(ref_segments: list, cmp_segments: list, threshold: float = 0.7) -> dict:
    """Greedy one-to-one segment alignment using visual embedding similarity.

    Args:
        ref_segments: Segments for reference video (from get_segments_for_video).
        cmp_segments: Segments for compare video.
        threshold: Minimum cosine similarity to consider two segments a match.
            Pairs below this value are treated as missing/added. Default 0.7.

    Returns:
        Dict with summary (matched, changed, missing, added, shifted, avg_shift,
        overall_similarity), language_variant, modality_similarity, and segments.
        Each segment entry includes a time_shift field (seconds, None for
        missing/added segments).
    """
    ref_by_seg = _group_by_segment(ref_segments)
    cmp_by_seg = _group_by_segment(cmp_segments)

    ref_ids = sorted(ref_by_seg.keys())
    cmp_ids = sorted(cmp_by_seg.keys())

    # Build similarity matrix using visual embeddings
    pairs = []
    for r_id in ref_ids:
        r_vis = ref_by_seg[r_id].get("visual", {}).get("embedding")
        if not r_vis:
            continue
        for c_id in cmp_ids:
            c_vis = cmp_by_seg[c_id].get("visual", {}).get("embedding")
            if not c_vis:
                continue
            sim = cosine_similarity(r_vis, c_vis)
            pairs.append((sim, r_id, c_id))

    # Greedy matching: highest similarity first
    pairs.sort(reverse=True)
    matched_ref = set()
    matched_cmp = set()
    alignments = []

    for sim, r_id, c_id in pairs:
        if r_id in matched_ref or c_id in matched_cmp:
            continue
        if sim < threshold:
            break

        modality_scores = {}
        for mod in ["visual", "audio", "transcription"]:
            r_emb = ref_by_seg[r_id].get(mod, {}).get("embedding")
            c_emb = cmp_by_seg[c_id].get(mod, {}).get("embedding")
            if r_emb and c_emb:
                modality_scores[mod] = round(cosine_similarity(r_emb, c_emb), 4)
            else:
                modality_scores[mod] = None

        ref_info = _segment_info(r_id, ref_by_seg[r_id])
        cmp_info = _segment_info(c_id, cmp_by_seg[c_id])
        time_shift = round(cmp_info["start_time"] - ref_info["start_time"], 2)

        status = "matched" if sim >= 0.9 else "changed"
        alignments.append({
            "status": status,
            "similarity": round(sim, 4),
            "time_shift": time_shift,
            "reference": ref_info,
            "compare": cmp_info,
            "modality_scores": modality_scores,
        })
        matched_ref.add(r_id)
        matched_cmp.add(c_id)

    # Missing segments (in reference, not matched)
    for r_id in ref_ids:
        if r_id not in matched_ref:
            alignments.append({
                "status": "missing",
                "similarity": None,
                "time_shift": None,
                "reference": _segment_info(r_id, ref_by_seg[r_id]),
                "compare": None,
                "modality_scores": None,
            })

    # Added segments (in compare, not matched)
    for c_id in cmp_ids:
        if c_id not in matched_cmp:
            alignments.append({
                "status": "added",
                "similarity": None,
                "time_shift": None,
                "reference": None,
                "compare": _segment_info(c_id, cmp_by_seg[c_id]),
                "modality_scores": None,
            })

    alignments.sort(key=_alignment_sort_key)

    # Compute summary
    counts = {"matched": 0, "changed": 0, "missing": 0, "added": 0}
    all_modality_sims = {"visual": [], "audio": [], "transcription": []}
    for a in alignments:
        counts[a["status"]] += 1
        if a["modality_scores"]:
            for mod, score in a["modality_scores"].items():
                if score is not None:
                    all_modality_sims[mod].append(score)

    modality_similarity = {}
    for mod, sims in all_modality_sims.items():
        modality_similarity[mod] = round(np.mean(sims), 4) if sims else 0.0

    matched_sims = [a["similarity"] for a in alignments if a["similarity"] is not None]
    overall_sim = round(np.mean(matched_sims), 4) if matched_sims else 0.0

    shifts = [a["time_shift"] for a in alignments if a["time_shift"] is not None and abs(a["time_shift"]) > 0.5]
    counts["shifted"] = len(shifts)
    counts["avg_shift"] = round(sum(abs(s) for s in shifts) / len(shifts), 2) if shifts else 0.0

    # Language variant detection
    lang_variant = {"detected": False}
    vis_avg = modality_similarity.get("visual", 0)
    aud_avg = modality_similarity.get("audio", 0)
    tra_avg = modality_similarity.get("transcription", 0)
    if vis_avg >= 0.9 and (aud_avg < 0.5 or tra_avg < 0.5):
        lang_variant = {
            "detected": True,
            "visual_similarity": vis_avg,
            "audio_similarity": aud_avg,
            "transcription_similarity": tra_avg,
        }

    return {
        "summary": {**counts, "overall_similarity": overall_sim},
        "language_variant": lang_variant,
        "modality_similarity": modality_similarity,
        "segments": alignments,
    }


def _group_by_segment(segments: list) -> dict:
    """Group segment embeddings by segment_id, then by modality."""
    grouped = {}
    for seg in segments:
        sid = seg.get("segment_id")
        if sid is None:
            continue
        if sid not in grouped:
            grouped[sid] = {}
        modality = seg.get("modality_type")
        if modality:
            grouped[sid][modality] = {
                "embedding": seg.get("embedding"),
                "start_time": seg.get("start_time", 0),
                "end_time": seg.get("end_time", 0),
                "s3_uri": seg.get("s3_uri", ""),
            }
    return grouped


def _segment_info(seg_id: int, seg_data: dict) -> dict:
    """Extract display info from grouped segment data."""
    for mod in ["visual", "audio", "transcription"]:
        if mod in seg_data:
            return {
                "segment_id": seg_id,
                "start_time": seg_data[mod].get("start_time", 0),
                "end_time": seg_data[mod].get("end_time", 0),
            }
    return {"segment_id": seg_id, "start_time": 0, "end_time": 0}


def _alignment_sort_key(a: dict) -> tuple:
    """Sort alignments: matched/changed by ref time, then missing, then added."""
    status_order = {"matched": 0, "changed": 0, "missing": 1, "added": 2}
    order = status_order.get(a["status"], 3)
    if a["reference"]:
        return (order, a["reference"].get("start_time", 999999))
    if a["compare"]:
        return (order, a["compare"].get("start_time", 999999))
    return (order, 999999)
