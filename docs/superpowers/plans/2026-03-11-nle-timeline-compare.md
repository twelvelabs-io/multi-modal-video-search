# NLE Timeline Compare View — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the table-based compare diff view with a stacked NLE timeline supporting N-video comparison, time-shift detection, and threshold-based highlighting.

**Architecture:** Backend extends `align_segments()` with time_shift field and configurable threshold, adds a new `/api/compare/multi-diff` endpoint that loops alignment for N videos. Frontend replaces the segment table in `static/index.html` with a timeline renderer, controls bar, consensus heatmap, and detail panel — all driven by a `timelineState` object for client-side re-rendering without API re-calls.

**Tech Stack:** Python/FastAPI backend, vanilla JS frontend (single HTML file), TwelveLabs design system CSS variables, existing `compare_client.py` alignment module.

**Spec:** `docs/superpowers/specs/2026-03-11-nle-timeline-compare-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/compare_client.py` | Modify (lines 110-230) | Add `time_shift`, `threshold` param, shift summary |
| `app.py` | Add endpoint after line 860 | Add `/api/compare/multi-diff` endpoint |
| `static/index.html` | Modify | Replace compare diff UI with NLE timeline (CSS + HTML + JS) |

No new files. All changes are within existing files following the project's established patterns.

---

## Chunk 1: Backend Changes

### Task 1: Add time_shift and configurable threshold to align_segments

**Files:**
- Modify: `src/compare_client.py:110-230`

This task modifies the `align_segments()` function to:
1. Accept a `threshold` parameter (default 0.7) instead of the hard-coded `0.7` cutoff
2. Compute `time_shift` for each matched/changed segment pair
3. Add `shifted` count and `avg_shift` to the summary

- [ ] **Step 1: Update function signature to accept threshold**

In `src/compare_client.py`, change line 110 from:

```python
def align_segments(ref_segments: list, cmp_segments: list) -> dict:
```

to:

```python
def align_segments(ref_segments: list, cmp_segments: list, threshold: float = 0.7) -> dict:
```

- [ ] **Step 2: Replace hard-coded 0.7 with threshold parameter**

In the greedy matching loop (around line 148), change:

```python
if sim < 0.7:
    break
```

to:

```python
if sim < threshold:
    break
```

- [ ] **Step 3: Add time_shift to matched/changed segment output**

In the greedy matching loop (around lines 150-169), refactor the alignment dict construction. Currently the code calls `_segment_info()` inline. Extract it to named variables and add the time_shift:

```python
ref_info = _segment_info(r_id, ref_by_seg[r_id])
cmp_info = _segment_info(c_id, cmp_by_seg[c_id])
time_shift = round(cmp_info["start_time"] - ref_info["start_time"], 2)

alignments.append({
    "status": status,
    "similarity": round(sim, 4),
    "time_shift": time_shift,
    "reference": ref_info,
    "compare": cmp_info,
    "modality_scores": modality_scores,
})
```

This replaces the existing `alignments.append({...})` block that currently calls `_segment_info()` inline.

- [ ] **Step 4: Add time_shift: null for missing and added segments**

In the missing segment block (around lines 171-180) and the added segment block (around lines 182-191), add `"time_shift": None` to each alignment dict.

- [ ] **Step 5: Add shifted count and avg_shift to summary**

After the existing summary computation (around lines 195-210), add:

Add to the `counts` dict (which gets unpacked into the summary at return time):

```python
# Count shifted segments (abs(time_shift) > 0.5s)
shifts = [a["time_shift"] for a in alignments if a["time_shift"] is not None and abs(a["time_shift"]) > 0.5]
counts["shifted"] = len(shifts)
counts["avg_shift"] = round(sum(abs(s) for s in shifts) / len(shifts), 2) if shifts else 0.0
```

Note: The existing code uses a `counts` dict that gets unpacked as `{**counts, "overall_similarity": overall_sim}` in the return statement. Add `shifted` and `avg_shift` to `counts` before that return.

- [ ] **Step 6: Verify manually**

Run the app locally and hit the existing `/api/compare/diff` endpoint with two known video IDs. Verify the response now includes `time_shift` on each segment and `shifted`/`avg_shift` in the summary.

```bash
cd /Users/bpenven/Documents/Code/multi-modal-video-search
python3 app.py &
# Then in another terminal:
curl -s -X POST http://localhost:8000/api/compare/diff \
  -H "Content-Type: application/json" \
  -d '{"reference_video_id":"<KNOWN_REF_ID>","compare_video_id":"<KNOWN_CMP_ID>"}' | python3 -m json.tool | head -30
```

Check that:
- Each segment in `segments` array has a `time_shift` field (number or null)
- `summary` has `shifted` (int) and `avg_shift` (float)

- [ ] **Step 7: Commit**

```bash
git add src/compare_client.py
git commit -m "feat(compare): add time_shift detection and configurable threshold to align_segments"
```

---

### Task 2: Add multi-diff API endpoint

**Files:**
- Modify: `app.py` (insert after the existing `/api/compare/diff` endpoint at ~line 860)

This task adds a new `POST /api/compare/multi-diff` endpoint that accepts a reference video ID and an array of comparison video IDs, runs `align_segments()` for each pair, and returns a structured response with all alignments.

- [ ] **Step 1: Add the multi-diff endpoint**

Insert after the existing `/api/compare/diff` endpoint (around line 860). The endpoint should:

```python
@app.post("/api/compare/multi-diff")
async def compare_multi_diff(request: Request):
    """Compare a reference video against multiple comparison videos simultaneously."""
    body = await request.json()
    reference_id = body.get("reference_id")
    compare_ids = body.get("compare_ids", [])

    if not reference_id or not compare_ids:
        return JSONResponse({"error": "reference_id and compare_ids required"}, status_code=400)

    if len(compare_ids) > 8:
        return JSONResponse({"error": "Maximum 8 comparison videos"}, status_code=400)

    # Get clients (same pattern as existing /api/compare/diff endpoint)
    search_client = get_search_client()
    mongodb = search_client.get_mongodb_client()

    # Fetch reference data
    ref_segments = mongodb.get_segments_for_video(reference_id)
    if not ref_segments:
        return JSONResponse({"error": f"No segments found for reference video {reference_id}"}, status_code=404)
    ref_fp = mongodb.get_video_fingerprint(reference_id)
    ref_metadata = {
        "name": ref_fp.get("video_name", reference_id) if ref_fp else reference_id,
        "duration": ref_fp.get("total_duration", 0) if ref_fp else 0,
        "segment_count": ref_fp.get("segment_count", 0) if ref_fp else 0,
    }
    ref_tech = _get_tech_metadata(reference_id, ref_fp)

    # Process each comparison video
    comparisons = []
    for cmp_id in compare_ids:
        try:
            cmp_segments = mongodb.get_segments_for_video(cmp_id)
            cmp_fp = mongodb.get_video_fingerprint(cmp_id)

            alignment = align_segments(ref_segments, cmp_segments)

            cmp_metadata = {
                "name": cmp_fp.get("video_name", cmp_id) if cmp_fp else cmp_id,
                "duration": cmp_fp.get("total_duration", 0) if cmp_fp else 0,
                "segment_count": cmp_fp.get("segment_count", 0) if cmp_fp else 0,
            }
            cmp_tech = _get_tech_metadata(cmp_id, cmp_fp)

            comparisons.append({
                "video_id": cmp_id,
                "metadata": cmp_metadata,
                "technical_metadata": cmp_tech,
                "alignment": alignment
            })
        except Exception as e:
            comparisons.append({
                "video_id": cmp_id,
                "error": str(e),
                "metadata": {"name": cmp_id},
                "technical_metadata": None,
                "alignment": None
            })

    # Resolve video URLs for all videos
    video_urls = {}
    all_ids = [reference_id] + compare_ids
    for vid_id in all_ids:
        try:
            s3_key = _resolve_s3_key(vid_id)
            if s3_key and CLOUDFRONT_DOMAIN:
                video_urls[vid_id] = f"https://{CLOUDFRONT_DOMAIN}/{s3_key}"
        except Exception:
            pass

    return {
        "reference": {
            "video_id": reference_id,
            "metadata": ref_metadata,
            "technical_metadata": ref_tech
        },
        "comparisons": comparisons,
        "video_urls": video_urls
    }
```

Note: Make sure `align_segments` is imported from `compare_client` at the top of `app.py` — it already is (check the existing `/api/compare/diff` endpoint for the import pattern).

- [ ] **Step 2: Verify manually**

```bash
curl -s -X POST http://localhost:8000/api/compare/multi-diff \
  -H "Content-Type: application/json" \
  -d '{"reference_id":"<REF_ID>","compare_ids":["<CMP_ID_1>","<CMP_ID_2>"]}' | python3 -m json.tool | head -50
```

Check that:
- `reference` object has `video_id`, `metadata`, `technical_metadata`
- `comparisons` is an array with one entry per compare_id
- Each comparison has `alignment.summary` with `shifted` and `avg_shift`
- Each comparison has `alignment.segments` with `time_shift` on each
- `video_urls` maps video IDs to CloudFront URLs

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(compare): add /api/compare/multi-diff endpoint for N-video comparison"
```

---

## Chunk 2: Frontend — CSS Styles

### Task 3: Add NLE timeline CSS to index.html

**Files:**
- Modify: `static/index.html` (insert CSS after the existing compare CSS block, around line 3700)

Add all timeline-specific styles. Insert these after the existing `.compare-report-*` styles.

- [ ] **Step 1: Add timeline layout and track styles**

Insert after the last compare CSS rule (around line 3700, after `.compare-report-bar .bar-fill`):

```css
/* ═══════════════════════════════════════════════════════════════
   NLE TIMELINE COMPARE VIEW
   ═══════════════════════════════════════════════════════════════ */

/* Video Selector Strip */
.tl-selector-strip {
    display: flex;
    gap: 8px;
    padding: 12px 0;
    overflow-x: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
}
.tl-video-card {
    flex: 0 0 116px;
    border: 2px solid var(--border-light);
    border-radius: 8px;
    background: var(--bg-elevated);
    padding: 8px;
    cursor: pointer;
    transition: all 0.15s;
    position: relative;
    text-align: center;
}
.tl-video-card:hover { border-color: var(--border-hover); }
.tl-video-card.selected { border-color: var(--accent); }
.tl-video-card .tl-badge {
    position: absolute;
    top: 4px;
    left: 4px;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 1px 5px;
    border-radius: 3px;
    letter-spacing: 0.5px;
}
.tl-badge.ref { background: var(--accent); color: var(--text-inverse); }
.tl-badge.num { background: var(--bg-surface); color: var(--text-secondary); border: 1px solid var(--border); }
.tl-badge.lang { background: #7C3AED; color: #fff; }
.tl-video-card .tl-card-name {
    font-size: 11px;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 4px;
}
.tl-video-card .tl-card-meta {
    font-size: 9px;
    color: var(--text-tertiary);
    font-family: 'IBM Plex Mono', monospace;
}

/* Controls Bar */
.tl-controls {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-light);
}
.tl-threshold-group {
    display: flex;
    align-items: center;
    gap: 8px;
}
.tl-threshold-group label {
    font-size: 11px;
    color: var(--text-secondary);
    white-space: nowrap;
}
.tl-threshold-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 140px;
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(to right, var(--error), var(--warning), var(--success));
    outline: none;
}
.tl-threshold-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--text-primary);
    cursor: pointer;
    border: 2px solid var(--bg-body);
}
.tl-threshold-slider::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--text-primary);
    cursor: pointer;
    border: 2px solid var(--bg-body);
}
.tl-threshold-val {
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-primary);
    min-width: 32px;
}
.tl-pills {
    display: flex;
    gap: 4px;
}
.tl-pill {
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
}
.tl-pill:hover { border-color: var(--border-hover); color: var(--text-primary); }
.tl-pill.active { background: var(--accent); color: var(--text-inverse); border-color: var(--accent); }
.tl-pills-right { margin-left: auto; }

/* Timeline Container */
.tl-timeline {
    position: relative;
    margin-top: 8px;
    user-select: none;
}

/* Ruler */
.tl-ruler {
    display: flex;
    align-items: flex-end;
    height: 24px;
    padding-left: 130px;
    border-bottom: 1px solid var(--border-light);
    position: relative;
}
.tl-ruler-tick {
    position: absolute;
    font-size: 9px;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-tertiary);
    transform: translateX(-50%);
    bottom: 2px;
}

/* Tracks */
.tl-track {
    display: flex;
    align-items: center;
    height: 36px;
    border-bottom: 1px solid var(--border-light);
}
.tl-track-label {
    flex: 0 0 130px;
    display: flex;
    align-items: center;
    gap: 6px;
    padding-right: 8px;
    overflow: hidden;
}
.tl-track-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}
.tl-track-name {
    font-size: 11px;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.tl-track-info {
    font-size: 9px;
    color: var(--text-tertiary);
    font-family: 'IBM Plex Mono', monospace;
    white-space: nowrap;
}
.tl-track-bar {
    flex: 1;
    display: flex;
    height: 28px;
    gap: 1px;
    position: relative;
}

/* Segments */
.tl-seg {
    height: 100%;
    min-width: 4px;
    border-radius: 2px;
    cursor: pointer;
    transition: filter 0.1s;
    position: relative;
}
.tl-seg:hover { filter: brightness(1.2); }
.tl-seg[data-tip]:hover::after {
    content: attr(data-tip);
    position: absolute;
    bottom: calc(100% + 4px);
    left: 50%;
    transform: translateX(-50%);
    font-size: 9px;
    font-family: 'IBM Plex Mono', monospace;
    background: var(--bg-body);
    color: var(--text-primary);
    padding: 2px 6px;
    border-radius: 3px;
    border: 1px solid var(--border);
    white-space: nowrap;
    z-index: 10;
    pointer-events: none;
}
.tl-seg.matched { background: rgba(96, 226, 27, 0.15); }
.tl-seg.changed { background: rgba(250, 186, 23, 0.15); }
.tl-seg.missing { background: rgba(226, 38, 34, 0.20); }
.tl-seg.added   { background: rgba(96, 226, 27, 0.15); border: 1px dashed var(--success); }
.tl-seg.shifted { border-left: 2px solid var(--warning); }
.tl-seg.selected { outline: 2px solid var(--accent); outline-offset: 1px; }

/* Alert animation for segments below threshold */
@keyframes tl-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(226, 38, 34, 0.4); }
    50% { box-shadow: 0 0 0 3px rgba(226, 38, 34, 0); }
}
.tl-seg.alert { animation: tl-pulse 1.5s infinite; outline: 1px solid var(--error); }

/* Playhead */
.tl-playhead {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 1px;
    background: var(--text-primary);
    pointer-events: none;
    z-index: 5;
    left: 130px; /* default, updated via JS */
}
.tl-playhead::before {
    content: '';
    position: absolute;
    top: -4px;
    left: -5px;
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid var(--text-primary);
}
.tl-playhead-label {
    position: absolute;
    top: -18px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 9px;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-primary);
    background: var(--bg-elevated);
    padding: 1px 4px;
    border-radius: 2px;
    white-space: nowrap;
}
/* Playhead drag handle (invisible, wider for easier grabbing) */
.tl-playhead-handle {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 11px;
    left: 125px; /* synced with playhead, updated via JS */
    cursor: col-resize;
    z-index: 6;
}

/* Consensus Heatmap */
.tl-consensus {
    display: flex;
    height: 12px;
    margin-left: 130px;
    gap: 1px;
    margin-top: 2px;
    margin-bottom: 4px;
}
.tl-cons-seg {
    height: 100%;
    border-radius: 1px;
    min-width: 2px;
}

/* Detail Panel */
.tl-detail {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-surface);
    margin-top: 8px;
    overflow: hidden;
    display: none;
}
.tl-detail.visible { display: block; }
.tl-detail-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-light);
}
.tl-detail-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    font-family: 'IBM Plex Mono', monospace;
}
.tl-detail-badges { display: flex; gap: 6px; }
.tl-detail-badge {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
}
.tl-detail-badge.error { background: rgba(226,38,34,0.15); color: var(--error); }
.tl-detail-badge.warn  { background: rgba(250,186,23,0.15); color: var(--warning); }

/* Frame Grid in Detail Panel */
.tl-frame-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 10px;
    padding: 12px 14px;
}
.tl-frame-card {
    border: 1px solid var(--border-light);
    border-radius: 6px;
    background: var(--bg-elevated);
    padding: 8px;
    cursor: pointer;
    transition: border-color 0.15s;
}
.tl-frame-card:hover { border-color: var(--border-hover); }
.tl-frame-card.focused { border-color: var(--accent); }
.tl-frame-card-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.tl-frame-card-thumb {
    width: 100%;
    aspect-ratio: 16/9;
    background: var(--bg-body);
    border-radius: 4px;
    margin-bottom: 6px;
    overflow: hidden;
}
.tl-frame-card-thumb img { width: 100%; height: 100%; object-fit: cover; }
.tl-frame-card-score {
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
}
.tl-frame-card-shift {
    font-size: 10px;
    color: var(--text-tertiary);
    font-family: 'IBM Plex Mono', monospace;
}

/* Modality Breakdown in Detail Panel */
.tl-modality-table {
    padding: 8px 14px 12px;
    display: grid;
    grid-template-columns: 80px repeat(auto-fill, minmax(80px, 1fr));
    gap: 4px 12px;
    font-size: 11px;
}
.tl-mod-header { color: var(--text-tertiary); font-weight: 600; }
.tl-mod-cell { color: var(--text-primary); font-family: 'IBM Plex Mono', monospace; }

/* Detail Panel Actions */
.tl-detail-actions {
    display: flex;
    gap: 8px;
    padding: 8px 14px 12px;
    border-top: 1px solid var(--border-light);
}
.tl-btn-analyze {
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid #7C3AED;
    background: rgba(124, 58, 237, 0.15);
    color: #C4B5FD;
    transition: all 0.15s;
}
.tl-btn-analyze:hover { background: #7C3AED; color: #fff; }
.tl-btn-ghost {
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 11px;
    cursor: pointer;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-secondary);
    transition: all 0.15s;
}
.tl-btn-ghost:hover { border-color: var(--border-hover); color: var(--text-primary); }

/* Stats Bar */
.tl-stats {
    display: flex;
    gap: 16px;
    padding: 8px 0;
    font-size: 12px;
    color: var(--text-secondary);
    border-top: 1px solid var(--border-light);
    margin-top: 4px;
}
.tl-stat-val { font-weight: 600; margin-right: 3px; }
.tl-stat-val.matched { color: var(--success); }
.tl-stat-val.changed { color: var(--warning); }
.tl-stat-val.missing { color: var(--error); }
.tl-stat-val.alert   { color: var(--error); }

/* Loading skeleton */
.tl-skeleton {
    height: 36px;
    background: linear-gradient(90deg, var(--bg-surface) 25%, var(--bg-elevated) 50%, var(--bg-surface) 75%);
    background-size: 200% 100%;
    animation: tl-shimmer 1.5s infinite;
    border-radius: 4px;
    margin-bottom: 2px;
}
@keyframes tl-shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* Empty state */
.tl-empty {
    text-align: center;
    padding: 48px 24px;
    color: var(--text-tertiary);
    font-size: 13px;
}

/* Multi-select result cards */
.compare-result-card.selected-for-compare {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
}
```

- [ ] **Step 2: Commit**

```bash
git add static/index.html
git commit -m "style(compare): add NLE timeline CSS for tracks, segments, controls, and detail panel"
```

---

## Chunk 3: Frontend — HTML Structure and JavaScript

### Task 4: Add timeline HTML structure inside compare diff container

**Files:**
- Modify: `static/index.html` (around lines 4553-4590 — the `#compareDiffContainer` area)

Replace the contents of the compare diff container with the NLE timeline HTML. The existing `#compareSegmentsTable`, `#compareStatsBar`, and related elements get replaced.

- [ ] **Step 1: Replace compareDiffContainer inner HTML**

Find the `#compareDiffContainer` div (around line 4553). Replace its contents (the stats bar, lang variant banner, tech meta panel, segments table) with the new timeline structure:

```html
<div id="compareDiffContainer" class="compare-diff-container" style="display:none;">
    <!-- Language Variant Banner (keep existing) -->
    <div id="langVariantBanner" class="lang-variant-banner">
        <strong>Language Variant Detected</strong> — <span id="langVariantText"></span>
    </div>

    <!-- Video Selector Strip -->
    <div id="tlSelectorStrip" class="tl-selector-strip"></div>

    <!-- Controls Bar -->
    <div class="tl-controls">
        <div class="tl-threshold-group">
            <label>Threshold</label>
            <input type="range" class="tl-threshold-slider" id="tlThreshold" min="0" max="100" value="85">
            <span class="tl-threshold-val" id="tlThresholdVal">85%</span>
        </div>
        <div class="tl-pills" id="tlModalityPills">
            <button class="tl-pill active" data-mode="combined" onclick="setTimelineModality(this)">Combined</button>
            <button class="tl-pill" data-mode="visual" onclick="setTimelineModality(this)">Visual</button>
            <button class="tl-pill" data-mode="audio" onclick="setTimelineModality(this)">Audio</button>
            <button class="tl-pill" data-mode="transcription" onclick="setTimelineModality(this)">Speech</button>
        </div>
        <div class="tl-pills tl-pills-right" id="tlViewPills">
            <button class="tl-pill active" data-view="all" onclick="setTimelineView(this)">All</button>
            <button class="tl-pill" data-view="diffs" onclick="setTimelineView(this)">Diffs Only</button>
            <button class="tl-pill" data-view="shifts" onclick="setTimelineView(this)">Shifts</button>
        </div>
    </div>

    <!-- Timeline Area -->
    <div class="tl-timeline" id="tlTimeline">
        <div class="tl-ruler" id="tlRuler"></div>
        <div id="tlTracks"></div>
        <div class="tl-consensus" id="tlConsensus"></div>
        <div class="tl-playhead" id="tlPlayhead" style="display:none;">
            <span class="tl-playhead-label" id="tlPlayheadLabel">0:00.0</span>
        </div>
        <div class="tl-playhead-handle" id="tlPlayheadHandle" style="display:none;"></div>
    </div>

    <!-- Detail Panel -->
    <div class="tl-detail" id="tlDetail">
        <div class="tl-detail-header">
            <span class="tl-detail-title" id="tlDetailTitle"></span>
            <div class="tl-detail-badges" id="tlDetailBadges"></div>
        </div>
        <div class="tl-frame-grid" id="tlFrameGrid"></div>
        <div class="tl-modality-table" id="tlModalityTable"></div>
        <div class="tl-detail-actions">
            <button class="tl-btn-analyze" id="tlAnalyzeBtn" onclick="tlAnalyzeSegment()">Analyze with AI</button>
            <button class="tl-btn-ghost" onclick="tlExportSegment()">Export Segment</button>
        </div>
        <div id="tlAnalysisResult" class="frame-analysis"></div>
    </div>

    <!-- Stats Bar -->
    <div class="tl-stats" id="tlStats"></div>
</div>
```

- [ ] **Step 2: Keep the report/export buttons in the compare header**

The existing compare header with Report/Export buttons should remain. Verify that `#compareDiffBtns` (the button group with Report, Export CSV, Export JSON) is still intact outside the diff container. If it's inside, move it above.

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat(compare): add NLE timeline HTML structure replacing segment table"
```

---

### Task 5: Add timeline state and rendering JavaScript

**Files:**
- Modify: `static/index.html` (in the `<script>` section, after existing compare functions around line 8335)

This is the core rendering logic. Add the timeline state object and all rendering functions.

- [ ] **Step 1: Add timeline state object and helper functions**

Insert after the existing compare JavaScript functions:

```javascript
// ═══════════════════════════════════════════════════════════
// NLE TIMELINE STATE & RENDERING
// ═══════════════════════════════════════════════════════════

const tlState = {
    threshold: 85,             // integer 0-100 matching slider value
    modalityMode: 'combined',
    viewFilter: 'all',
    selectedSegmentIdx: null,
    selectedCompareIdx: 0,     // which compare video is focused in detail panel
    includedVideoIds: [],      // which compare videos are toggled on
    playheadTime: 0,
    apiResponse: null,         // cached multi-diff response
    refDuration: 0,            // reference video duration in seconds
};

// Named tlFormatTime to avoid collision with existing tlFormatTime() global
function tlFormatTime(seconds) {
    if (seconds == null) return '--:--';
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(1);
    return `${m}:${s.padStart(4, '0')}`;
}

const TL_DOT_COLORS = [
    'var(--visual)', 'var(--audio)', 'var(--transcription)',
    'var(--accent)', '#FFB592', '#C084FC', '#60A5FA', '#FF6B9D'
];

function getSegScore(seg) {
    // Get the similarity score based on current modality mode
    if (!seg || seg.similarity == null) return null;
    if (tlState.modalityMode === 'combined') return seg.similarity;
    const ms = seg.modality_scores;
    if (!ms) return seg.similarity;
    return ms[tlState.modalityMode] ?? seg.similarity;
}

function segColorClass(seg) {
    // Return CSS class for segment status + modifiers
    const classes = [seg.status];
    if (seg.time_shift != null && Math.abs(seg.time_shift) > 0.5) classes.push('shifted');
    const score = getSegScore(seg);
    if (score != null && score < tlState.threshold / 100) classes.push('alert');
    return classes.join(' ');
}

function shouldShowSeg(seg) {
    if (tlState.viewFilter === 'all') return true;
    if (tlState.viewFilter === 'diffs') return seg.status !== 'matched';
    if (tlState.viewFilter === 'shifts') return seg.time_shift != null && Math.abs(seg.time_shift) > 0.5;
    return true;
}

function statusColor(status) {
    const map = { matched: 'var(--success)', changed: 'var(--warning)', missing: 'var(--error)', added: 'var(--success)' };
    return map[status] || 'var(--text-tertiary)';
}
```

- [ ] **Step 2: Add showTimelineDiff function (replaces showCompareDiff flow)**

This function is called when the user clicks a result card. It replaces the old `showCompareDiff` for multi-video mode:

```javascript
async function showTimelineDiff(refId, compareIds, resultCards) {
    // Show the diff container, hide the results list
    const diffContainer = document.getElementById('compareDiffContainer');
    const resultsList = document.getElementById('compareResultsList');
    resultsList.style.display = 'none';
    diffContainer.style.display = 'block';

    // Show loading skeletons
    const tracksDiv = document.getElementById('tlTracks');
    tracksDiv.innerHTML = '';
    for (let i = 0; i < compareIds.length + 1; i++) {
        tracksDiv.innerHTML += '<div class="tl-skeleton"></div>';
    }
    document.getElementById('tlStats').innerHTML = '';
    document.getElementById('tlConsensus').innerHTML = '';
    document.getElementById('tlDetail').classList.remove('visible');

    try {
        const resp = await fetch('/api/compare/multi-diff', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reference_id: refId, compare_ids: compareIds })
        });
        const data = await resp.json();
        tlState.apiResponse = data;
        tlState.refDuration = data.reference.metadata.duration || 0;
        tlState.selectedSegmentIdx = null;

        // Store video URLs
        compareVideoUrls = data.video_urls || {};

        // Render selector strip
        renderTlSelectorStrip(data);

        // Render timeline
        renderTimeline();
    } catch (err) {
        tracksDiv.innerHTML = `<div class="tl-empty">Error loading comparison: ${err.message}</div>`;
    }
}
```

- [ ] **Step 3: Add video selector strip renderer**

```javascript
function renderTlSelectorStrip(data) {
    const strip = document.getElementById('tlSelectorStrip');
    strip.innerHTML = '';

    // Initialize includedVideoIds with all compare videos
    tlState.includedVideoIds = data.comparisons.map(c => c.video_id);

    // Reference card (always shown, not toggleable)
    const refCard = document.createElement('div');
    refCard.className = 'tl-video-card selected';
    refCard.innerHTML = `
        <span class="tl-badge ref">REF</span>
        <div class="tl-card-name">${data.reference.metadata.name || data.reference.video_id}</div>
        <div class="tl-card-meta">${data.reference.metadata.segment_count || 0} segs</div>
    `;
    strip.appendChild(refCard);

    // Compare video cards (clickable to toggle inclusion)
    data.comparisons.forEach((cmp, idx) => {
        const card = document.createElement('div');
        const isIncluded = tlState.includedVideoIds.includes(cmp.video_id);
        card.className = 'tl-video-card' + (isIncluded ? ' selected' : '');
        const lang = cmp.alignment?.language_variant?.detected ? '<span class="tl-badge lang">LANG</span>' : '';
        card.innerHTML = `
            <span class="tl-badge num">${idx + 2}</span>${lang}
            <div class="tl-card-name">${cmp.metadata.name || cmp.video_id}</div>
            <div class="tl-card-meta">${cmp.alignment ? cmp.alignment.summary.matched + cmp.alignment.summary.changed + cmp.alignment.summary.missing + cmp.alignment.summary.added : 0} segs</div>
        `;
        card.onclick = () => {
            const i = tlState.includedVideoIds.indexOf(cmp.video_id);
            if (i >= 0) { tlState.includedVideoIds.splice(i, 1); card.classList.remove('selected'); }
            else { tlState.includedVideoIds.push(cmp.video_id); card.classList.add('selected'); }
            renderTimeline(); // re-render with filtered comparisons
        };
        strip.appendChild(card);
    });

    // Language variant banner
    const hasLang = data.comparisons.some(c => c.alignment?.language_variant?.detected);
    const banner = document.getElementById('langVariantBanner');
    if (hasLang && banner) {
        banner.classList.add('visible');
        const langText = document.getElementById('langVariantText');
        if (langText) langText.textContent = 'High visual similarity with audio/transcription divergence detected in one or more videos.';
    } else if (banner) {
        banner.classList.remove('visible');
    }
}
```

- [ ] **Step 4: Add main renderTimeline function**

```javascript
function renderTimeline() {
    const data = tlState.apiResponse;
    if (!data) return;

    const tracksDiv = document.getElementById('tlTracks');
    const duration = tlState.refDuration || 1;
    tracksDiv.innerHTML = '';

    // Render ruler
    renderTlRuler(duration);

    // Reference track — use first comparison's segments to show ref side
    const refTrack = buildRefTrack(data);
    tracksDiv.appendChild(refTrack);

    // Comparison tracks (filtered by included videos)
    const visibleComparisons = data.comparisons.filter(c => tlState.includedVideoIds.includes(c.video_id));
    visibleComparisons.forEach((cmp, cmpIdx) => {
        if (cmp.error) {
            const errTrack = document.createElement('div');
            errTrack.className = 'tl-track';
            errTrack.innerHTML = `
                <div class="tl-track-label"><span class="tl-track-name" style="color:var(--error)">${cmp.metadata.name || cmp.video_id} — Error</span></div>
                <div class="tl-track-bar"><div class="tl-empty" style="font-size:10px;padding:4px">${cmp.error}</div></div>
            `;
            tracksDiv.appendChild(errTrack);
            return;
        }
        const track = buildCmpTrack(cmp, cmpIdx, duration);
        tracksDiv.appendChild(track);
    });

    // Consensus heatmap
    renderTlConsensus(data, duration);

    // Stats bar
    renderTlStats(data);

    // Show playhead
    const playhead = document.getElementById('tlPlayhead');
    const handle = document.getElementById('tlPlayheadHandle');
    playhead.style.display = 'block';
    handle.style.display = 'block';
    updatePlayheadPosition();
}
```

- [ ] **Step 5: Add track builder functions**

```javascript
function buildRefTrack(data) {
    const track = document.createElement('div');
    track.className = 'tl-track';
    const duration = tlState.refDuration || 1;

    // Gather unique reference segments from all comparisons
    const refSegs = new Map();
    data.comparisons.forEach(cmp => {
        if (!cmp.alignment) return;
        cmp.alignment.segments.forEach(seg => {
            if (seg.reference && !refSegs.has(seg.reference.segment_id)) {
                refSegs.set(seg.reference.segment_id, seg.reference);
            }
        });
    });
    const sortedSegs = [...refSegs.values()].sort((a, b) => a.start_time - b.start_time);

    track.innerHTML = `
        <div class="tl-track-label">
            <span class="tl-track-dot" style="background:var(--accent)"></span>
            <span class="tl-track-name">${data.reference.metadata.name || 'REF'}</span>
        </div>
    `;

    const bar = document.createElement('div');
    bar.className = 'tl-track-bar';
    sortedSegs.forEach(seg => {
        const segDur = seg.end_time - seg.start_time;
        const pct = (segDur / duration) * 100;
        const block = document.createElement('div');
        block.className = 'tl-seg matched';
        block.style.flexBasis = pct + '%';
        block.setAttribute('data-tip', `Seg ${seg.segment_id}`);
        bar.appendChild(block);
    });
    track.appendChild(bar);
    return track;
}

function buildCmpTrack(cmp, cmpIdx, duration) {
    const track = document.createElement('div');
    track.className = 'tl-track';

    const dotColor = TL_DOT_COLORS[cmpIdx % TL_DOT_COLORS.length];
    track.innerHTML = `
        <div class="tl-track-label">
            <span class="tl-track-dot" style="background:${dotColor}"></span>
            <span class="tl-track-name">${cmp.metadata.name || cmp.video_id}</span>
        </div>
    `;

    const bar = document.createElement('div');
    bar.className = 'tl-track-bar';

    // Build segments from alignment, using ref segment boundaries for positioning
    const segments = cmp.alignment.segments;
    segments.forEach((seg, segIdx) => {
        if (!shouldShowSeg(seg)) return;

        const refPart = seg.reference || seg.compare;
        if (!refPart) return;
        const segDur = refPart.end_time - refPart.start_time;
        const pct = (segDur / duration) * 100;

        const block = document.createElement('div');
        block.className = 'tl-seg ' + segColorClass(seg);
        block.style.flexBasis = pct + '%';

        if (segIdx === tlState.selectedSegmentIdx) block.classList.add('selected');

        const score = getSegScore(seg);
        const tipText = seg.status === 'missing' ? 'MISSING' :
                        seg.status === 'added' ? 'ADDED' :
                        `Seg ${(seg.reference || seg.compare).segment_id} · ${score != null ? Math.round(score * 100) + '%' : '--'}`;
        block.setAttribute('data-tip', tipText);

        block.onclick = () => selectTimelineSegment(segIdx, cmpIdx);
        bar.appendChild(block);
    });

    track.appendChild(bar);
    return track;
}
```

- [ ] **Step 6: Add ruler, consensus, and stats renderers**

```javascript
function renderTlRuler(duration) {
    const ruler = document.getElementById('tlRuler');
    ruler.innerHTML = '';
    const LABEL_W = 130;
    // Place tick marks every 30s (or adjust based on duration)
    const interval = duration > 300 ? 60 : duration > 60 ? 30 : 10;
    // Use pixel-based positioning to align with track bar flex layout
    const rulerWidth = ruler.offsetWidth || 800;
    const trackWidth = rulerWidth - LABEL_W;
    for (let t = 0; t <= duration; t += interval) {
        const px = LABEL_W + (t / duration) * trackWidth;
        const tick = document.createElement('span');
        tick.className = 'tl-ruler-tick';
        tick.style.left = px + 'px';
        tick.textContent = tlFormatTime(t);
        ruler.appendChild(tick);
    }
}

function renderTlConsensus(data, duration) {
    const cons = document.getElementById('tlConsensus');
    cons.innerHTML = '';

    // Use first comparison's alignment as reference segment grid
    const firstAlign = data.comparisons.find(c => c.alignment);
    if (!firstAlign) return;

    // For each reference segment, find min similarity across all comparisons
    const refSegs = firstAlign.alignment.segments.filter(s => s.reference);
    refSegs.forEach(refSeg => {
        if (!refSeg.reference) return;
        const segDur = refSeg.reference.end_time - refSeg.reference.start_time;
        const pct = (segDur / duration) * 100;

        let minSim = 1.0;
        let worstStatus = 'matched';
        data.comparisons.forEach(cmp => {
            if (!cmp.alignment) return;
            const matching = cmp.alignment.segments.find(s =>
                s.reference && s.reference.segment_id === refSeg.reference.segment_id
            );
            if (!matching) { minSim = 0; worstStatus = 'missing'; return; }
            const score = getSegScore(matching);
            if (score != null && score < minSim) {
                minSim = score;
                worstStatus = matching.status;
            }
        });

        const block = document.createElement('div');
        block.className = 'tl-cons-seg';
        block.style.flexBasis = pct + '%';
        block.style.background = statusColor(worstStatus);
        block.style.opacity = Math.max(0.3, minSim);
        cons.appendChild(block);
    });
}

function renderTlStats(data) {
    const statsDiv = document.getElementById('tlStats');

    // Aggregate across all comparisons
    let total = 0, matched = 0, changed = 0, missing = 0, belowThreshold = 0;
    data.comparisons.forEach(cmp => {
        if (!cmp.alignment) return;
        const s = cmp.alignment.summary;
        total += s.matched + s.changed + s.missing + s.added;
        matched += s.matched;
        changed += s.changed;
        missing += s.missing;
        // Count below threshold client-side
        cmp.alignment.segments.forEach(seg => {
            const score = getSegScore(seg);
            if (score != null && score < tlState.threshold / 100) belowThreshold++;
        });
    });

    statsDiv.innerHTML = `
        <span>Total: <span class="tl-stat-val">${total}</span></span>
        <span>Matched: <span class="tl-stat-val matched">${matched}</span></span>
        <span>Changed: <span class="tl-stat-val changed">${changed}</span></span>
        <span>Missing: <span class="tl-stat-val missing">${missing}</span></span>
        <span>Below threshold: <span class="tl-stat-val alert">${belowThreshold}</span></span>
    `;
}
```

- [ ] **Step 7: Commit**

```bash
git add static/index.html
git commit -m "feat(compare): add NLE timeline rendering JS — state, tracks, consensus, stats"
```

---

### Task 6: Add timeline interactivity — controls, playhead, detail panel

**Files:**
- Modify: `static/index.html` (append to the JavaScript added in Task 5)

- [ ] **Step 1: Add control handlers (threshold, modality, view filter)**

```javascript
// Controls
function setTimelineModality(btn) {
    tlState.modalityMode = btn.dataset.mode;
    document.querySelectorAll('#tlModalityPills .tl-pill').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    renderTimeline();
}

function setTimelineView(btn) {
    tlState.viewFilter = btn.dataset.view;
    document.querySelectorAll('#tlViewPills .tl-pill').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    renderTimeline();
}

// Threshold slider
document.addEventListener('DOMContentLoaded', () => {
    const slider = document.getElementById('tlThreshold');
    const valLabel = document.getElementById('tlThresholdVal');
    if (slider) {
        slider.addEventListener('input', () => {
            tlState.threshold = parseInt(slider.value);
            valLabel.textContent = slider.value + '%';
            renderTimeline();
        });
    }
});
```

- [ ] **Step 2: Add playhead drag logic**

```javascript
// Playhead
function updatePlayheadPosition() {
    const LABEL_W = 130;
    const duration = tlState.refDuration || 1;
    const playhead = document.getElementById('tlPlayhead');
    const handle = document.getElementById('tlPlayheadHandle');
    const label = document.getElementById('tlPlayheadLabel');
    if (!playhead) return;
    // Use pixel positioning to match track bar coordinate system
    const timeline = document.getElementById('tlTimeline');
    const totalWidth = timeline?.offsetWidth || 800;
    const trackWidth = totalWidth - LABEL_W;
    const px = LABEL_W + (tlState.playheadTime / duration) * trackWidth;
    playhead.style.left = px + 'px';
    handle.style.left = (px - 5) + 'px';
    label.textContent = tlFormatTime(tlState.playheadTime);
}

(function initPlayheadDrag() {
    let dragging = false;
    document.addEventListener('mousedown', e => {
        if (e.target.id === 'tlPlayheadHandle' || e.target.closest('#tlPlayheadHandle')) {
            dragging = true;
            e.preventDefault();
        }
    });
    document.addEventListener('mousemove', e => {
        if (!dragging) return;
        const timeline = document.getElementById('tlTimeline');
        if (!timeline) return;
        const rect = timeline.getBoundingClientRect();
        const trackStart = 130; // label width in px
        const trackWidth = rect.width - trackStart;
        const x = e.clientX - rect.left - trackStart;
        const pct = Math.max(0, Math.min(1, x / trackWidth));
        tlState.playheadTime = pct * (tlState.refDuration || 0);
        updatePlayheadPosition();
    });
    document.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false;
        // Snap to nearest segment boundary
        snapPlayheadToSegment();
    });
})();

function snapPlayheadToSegment() {
    const data = tlState.apiResponse;
    if (!data) return;
    const firstAlign = data.comparisons.find(c => c.alignment);
    if (!firstAlign) return;

    let closest = null, closestDist = Infinity;
    firstAlign.alignment.segments.forEach((seg, idx) => {
        if (!seg.reference) return;
        const dist = Math.abs(seg.reference.start_time - tlState.playheadTime);
        if (dist < closestDist) { closestDist = dist; closest = idx; }
    });
    if (closest != null) {
        const seg = firstAlign.alignment.segments[closest];
        if (seg.reference) {
            tlState.playheadTime = seg.reference.start_time;
            updatePlayheadPosition();
        }
    }
}
```

- [ ] **Step 3: Add segment selection and detail panel rendering**

```javascript
function selectTimelineSegment(segIdx, cmpIdx) {
    tlState.selectedSegmentIdx = segIdx;
    tlState.selectedCompareIdx = cmpIdx || 0;
    renderTimeline(); // re-render to update selected state
    renderTlDetail();
}

function renderTlDetail() {
    const data = tlState.apiResponse;
    if (!data || tlState.selectedSegmentIdx == null) {
        document.getElementById('tlDetail').classList.remove('visible');
        return;
    }

    const panel = document.getElementById('tlDetail');
    panel.classList.add('visible');

    // Find the selected segment across comparisons
    const segIdx = tlState.selectedSegmentIdx;
    const firstCmp = data.comparisons.find(c => c.alignment);
    if (!firstCmp) return;
    const refSeg = firstCmp.alignment.segments[segIdx];
    if (!refSeg) return;

    // Header
    const refPart = refSeg.reference;
    const title = refPart
        ? `Segment ${refPart.segment_id} — ${tlFormatTime(refPart.start_time)} → ${tlFormatTime(refPart.end_time)}`
        : `Added Segment`;
    document.getElementById('tlDetailTitle').textContent = title;

    // Badges
    let belowCount = 0, shiftCount = 0;
    data.comparisons.forEach(cmp => {
        if (!cmp.alignment) return;
        const seg = cmp.alignment.segments[segIdx];
        if (!seg) return;
        const score = getSegScore(seg);
        if (score != null && score < tlState.threshold / 100) belowCount++;
        if (seg.time_shift != null && Math.abs(seg.time_shift) > 0.5) shiftCount++;
    });
    const badges = document.getElementById('tlDetailBadges');
    badges.innerHTML = '';
    if (belowCount > 0) badges.innerHTML += `<span class="tl-detail-badge error">${belowCount} below threshold</span>`;
    if (shiftCount > 0) badges.innerHTML += `<span class="tl-detail-badge warn">${shiftCount} shifted</span>`;

    // Frame grid — one card per video
    const grid = document.getElementById('tlFrameGrid');
    grid.innerHTML = '';

    // Reference card
    const refCard = document.createElement('div');
    refCard.className = 'tl-frame-card';
    refCard.innerHTML = `
        <div class="tl-frame-card-label" style="color:var(--accent)">REF</div>
        <div class="tl-frame-card-thumb"></div>
        <div class="tl-frame-card-score">Reference</div>
    `;
    grid.appendChild(refCard);

    // Compare cards
    data.comparisons.forEach((cmp, cmpIdx) => {
        if (!cmp.alignment) return;
        const seg = cmp.alignment.segments[segIdx];
        if (!seg) return;

        const score = getSegScore(seg);
        const scoreText = score != null ? Math.round(score * 100) + '%' : '--';
        const shiftText = seg.time_shift != null && Math.abs(seg.time_shift) > 0
            ? (seg.time_shift > 0 ? '+' : '') + seg.time_shift.toFixed(1) + 's'
            : '';
        const dotColor = TL_DOT_COLORS[cmpIdx % TL_DOT_COLORS.length];

        const card = document.createElement('div');
        card.className = 'tl-frame-card' + (cmpIdx === tlState.selectedCompareIdx ? ' focused' : '');
        card.onclick = () => { tlState.selectedCompareIdx = cmpIdx; renderTlDetail(); };
        card.innerHTML = `
            <div class="tl-frame-card-label" style="color:${dotColor}">${cmp.metadata.name || cmp.video_id}</div>
            <div class="tl-frame-card-thumb"></div>
            <div class="tl-frame-card-score" style="color:${statusColor(seg.status)}">${scoreText}</div>
            ${shiftText ? `<div class="tl-frame-card-shift">${shiftText} shift</div>` : ''}
        `;
        grid.appendChild(card);
    });

    // Modality breakdown table (build HTML string then assign once)
    const modTable = document.getElementById('tlModalityTable');
    let modHtml = '<div class="tl-mod-header"></div>';
    data.comparisons.forEach(cmp => {
        modHtml += `<div class="tl-mod-header">${cmp.metadata.name || cmp.video_id}</div>`;
    });
    ['visual', 'audio', 'transcription'].forEach(mod => {
        modHtml += `<div class="tl-mod-header" style="text-transform:capitalize">${mod}</div>`;
        data.comparisons.forEach(cmp => {
            if (!cmp.alignment) { modHtml += '<div class="tl-mod-cell">--</div>'; return; }
            const seg = cmp.alignment.segments[segIdx];
            const val = seg?.modality_scores?.[mod];
            modHtml += `<div class="tl-mod-cell">${val != null ? Math.round(val * 100) + '%' : '--'}</div>`;
        });
    });
    modTable.innerHTML = modHtml;

    // Update analyze button state
    const analyzeBtn = document.getElementById('tlAnalyzeBtn');
    const focusedCmp = data.comparisons[tlState.selectedCompareIdx];
    const focusedSeg = focusedCmp?.alignment?.segments[segIdx];
    analyzeBtn.disabled = !focusedSeg || focusedSeg.status === 'missing' || focusedSeg.status === 'added';
    document.getElementById('tlAnalysisResult').innerHTML = '';
}
```

- [ ] **Step 4: Add Analyze with AI handler**

```javascript
async function tlAnalyzeSegment() {
    const data = tlState.apiResponse;
    if (!data || tlState.selectedSegmentIdx == null) return;

    const cmp = data.comparisons[tlState.selectedCompareIdx];
    if (!cmp?.alignment) return;
    const seg = cmp.alignment.segments[tlState.selectedSegmentIdx];
    if (!seg || !seg.reference || !seg.compare) return;

    const btn = document.getElementById('tlAnalyzeBtn');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';
    const resultDiv = document.getElementById('tlAnalysisResult');
    resultDiv.innerHTML = '<div style="padding:12px;color:var(--text-secondary);font-size:12px">Analyzing segment with AI...</div>';

    try {
        const resp = await fetch('/api/compare/analyze-segment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                reference_video_id: data.reference.video_id,
                compare_video_id: cmp.video_id,
                ref_start: seg.reference.start_time,
                ref_end: seg.reference.end_time,
                cmp_start: seg.compare.start_time,
                cmp_end: seg.compare.end_time
            })
        });
        const result = await resp.json();

        let html = `<div style="padding:8px 14px;font-size:11px;color:var(--text-secondary)">${result.total_frames} frames analyzed, ${result.differences_found} differences found</div>`;
        result.frames.forEach(f => {
            html += `<div class="frame-pair">
                <div><div class="frame-label">Reference @ ${f.timestamp.toFixed(1)}s</div><img src="data:image/jpeg;base64,${f.ref_image}" style="width:100%;border-radius:4px"></div>
                <div><div class="frame-label">Compare @ ${f.timestamp.toFixed(1)}s</div><img src="data:image/jpeg;base64,${f.cmp_image}" style="width:100%;border-radius:4px"></div>
                <span class="frame-diff-badge ${f.identical ? 'identical' : 'different'}">${f.identical ? 'Identical' : 'Different'}</span>
                ${f.difference ? `<div class="frame-diff-message">${f.difference}</div>` : ''}
            </div>`;
        });
        resultDiv.innerHTML = html;
    } catch (err) {
        resultDiv.innerHTML = `<div style="padding:12px;color:var(--error);font-size:12px">Analysis failed: ${err.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze with AI';
    }
}

function tlExportSegment() {
    const data = tlState.apiResponse;
    if (!data || tlState.selectedSegmentIdx == null) return;
    const cmp = data.comparisons[tlState.selectedCompareIdx];
    if (!cmp?.alignment) return;
    const seg = cmp.alignment.segments[tlState.selectedSegmentIdx];
    const blob = new Blob([JSON.stringify(seg, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `segment-${seg.reference?.segment_id || 'unknown'}.json`;
    a.click();
    URL.revokeObjectURL(url);
}
```

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat(compare): add timeline interactivity — controls, playhead, detail panel, AI analysis"
```

---

### Task 7: Wire up multi-video selection flow from find-similar results

**Files:**
- Modify: `static/index.html` (modify the existing `findSimilarVideos` function and result card click handlers, around lines 7733-7850)

Currently, clicking a result card calls `showCompareDiff(refId, cmpId)` for 2-video comparison. Change this to allow selecting multiple result cards, then trigger the timeline view.

- [ ] **Step 1: Add a "Compare Selected" button and multi-select state**

After the `#compareFindBtn` button (around line 4548), add a new button:

```html
<button id="compareSelectedBtn" class="compare-find-btn" style="display:none;margin-left:8px" onclick="compareSelectedVideos()">Compare Selected (0)</button>
```

Add a global to track selected compare videos:

```javascript
let compareSelectedIds = [];
```

- [ ] **Step 2: Modify result card click to toggle selection**

In the `findSimilarVideos()` function, find the result card creation loop (around line 7799 in the `results.forEach((r, idx) => { ... })` block). The variable `r.video_id` holds the compare video ID. Replace the existing `card.onclick` that calls `showCompareDiff(...)` with:

```javascript
const cmpId = r.video_id;
card.onclick = () => {
    const idx = compareSelectedIds.indexOf(cmpId);
    if (idx >= 0) {
        compareSelectedIds.splice(idx, 1);
        card.classList.remove('selected-for-compare');
    } else {
        if (compareSelectedIds.length >= 8) return; // cap at 8
        compareSelectedIds.push(cmpId);
        card.classList.add('selected-for-compare');
    }
    const btn = document.getElementById('compareSelectedBtn');
    btn.style.display = compareSelectedIds.length > 0 ? 'inline-block' : 'none';
    btn.textContent = `Compare Selected (${compareSelectedIds.length})`;
};
// Double-click for quick single-video compare
card.ondblclick = () => {
    showTimelineDiff(compareRefVideoId, [cmpId], [card]);
};
```

Note: `cmpId` is declared with `const` in the loop body (outside the closures) so both `onclick` and `ondblclick` share the same closure variable correctly.

- [ ] **Step 3: Add compareSelectedVideos function**

```javascript
function compareSelectedVideos() {
    if (compareSelectedIds.length === 0) return;
    showTimelineDiff(compareRefVideoId, compareSelectedIds, []);
}
```

- [ ] **Step 4: Redirect showCompareDiff to use the timeline**

The old `showCompareDiff(refId, cmpId, refName, cmpName)` function (around line 7955) references DOM elements that no longer exist (`#diffRefName`, `#diffCmpName`, `#compareSegmentsTable`, `#compareStatsBar`). It's still called by the Report panel and potentially other code paths. Replace the function body with a redirect to the timeline:

```javascript
function showCompareDiff(refId, cmpId, refName, cmpName) {
    // Redirect to NLE timeline view
    showTimelineDiff(refId, [cmpId], []);
}
```

This preserves backward compatibility for any remaining callers while routing everything through the new timeline.

Also update `showCompareReport()` (around line 8100) to work with `tlState.apiResponse` — replace references to `compareDiffData` with the timeline state. The simplest fix: at the top of `showCompareReport()`, add:

```javascript
if (!compareDiffData && tlState.apiResponse) {
    // Build legacy-format data from the first comparison for the report panel
    const first = tlState.apiResponse.comparisons[tlState.selectedCompareIdx || 0];
    if (first?.alignment) {
        compareDiffData = first.alignment;
        compareDiffData.reference_metadata = tlState.apiResponse.reference.technical_metadata;
        compareDiffData.compare_metadata = first.technical_metadata;
    }
}
```

- [ ] **Step 5: Reset selection when finding new similar videos**

At the top of `findSimilarVideos()`, add:

```javascript
compareSelectedIds = [];
document.getElementById('compareSelectedBtn').style.display = 'none';
```

- [ ] **Step 6: Verify end-to-end**

Run the app locally:
1. Go to Compare page
2. Select a reference video
3. Click "Find Similar"
4. Click (single click) multiple result cards — they should get accent border
5. "Compare Selected (N)" button appears
6. Click it — NLE timeline should load with all selected videos as tracks
7. Double-click a result card — should immediately show timeline with just that one video
8. Verify: tracks render, segments are colored, threshold slider works, clicking a segment opens detail panel

- [ ] **Step 7: Commit**

```bash
git add static/index.html
git commit -m "feat(compare): wire multi-video selection flow to NLE timeline"
```

---

## Post-Implementation

After all tasks are complete:
1. Push to `feature/nav-redesign-compare-upload` branch
2. Deploy to staging: `aws apprunner start-deployment --service-arn "arn:aws:apprunner:us-east-1:026090552520:service/video-search-staging/f5e755cb7c304fe4ba1b18a0b6154992" --region us-east-1`
3. Verify on staging with real videos
