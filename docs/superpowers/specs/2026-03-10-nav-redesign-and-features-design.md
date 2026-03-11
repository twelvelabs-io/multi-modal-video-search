# Navigation Redesign, Compare Page, and Upload Page

**Date:** 2026-03-10
**Status:** Approved

## Overview

Redesign the sidebar navigation with new labels and organization, add a Compare page for video-to-video QC/duplicate detection (inspired by WBD QC Genie project), add an Upload page for direct browser-based video ingestion with full processing control, and fold Examples into Search as suggestion chips.

## Navigation Structure

### Before

```
[Primary nav]
  Home          → search page
  Examples      → hardcoded sample queries
  Indexes       → video management

[EXPLORE]
  Analyze       → Claude chat + Pegasus
```

### After

```
[DISCOVER]
  Search        → renamed from Home, magnifier icon (Strand)
  Compare       → NEW, split-view columns icon (Strand)
  Analyze       → unchanged, magnifier+bars icon (Strand)

[MANAGE]
  Upload        → NEW, plus icon (Strand)
  Library       → renamed from Indexes, folder icon (Strand)
```

### Changes

| Old | New | Notes |
|-----|-----|-------|
| Home | Search | Direct label, Strand magnifier icon |
| Examples | *(removed)* | Folded into Search as suggestion chips |
| Indexes | Library | User-friendly name, same folder icon |
| *(new)* | Compare | Video-to-video comparison & QC |
| *(new)* | Upload | Direct browser upload with settings |
| Analyze | Analyze | Unchanged |
| Section: *(none)* + EXPLORE | DISCOVER + MANAGE | Two clear purpose-driven groups |

### Icons (all Strand SVGs)

- **Search:** `viewBox="0 0 12 11.707"` — magnifier (from existing search button)
- **Compare:** `viewBox="0 0 12 12"` — side-by-side columns/split-view
- **Analyze:** `viewBox="0 0 12 12"` — magnifier with bars (existing)
- **Upload:** `viewBox="0 0 12 12"` — plus sign
- **Library:** `viewBox="0 0 12 10"` — folder (existing Indexes icon)

## Search Page Changes

- Rename "Home" to "Search" in nav item and any page titles
- Add example query suggestion chips (from current Examples page content) displayed on the Search page when no search has been performed
- Remove the Examples page and its nav item
- Update `data-page="home"` to `data-page="search"` throughout

## Compare Page

### Context (WBD QC Genie)

Warner Bros. Discovery's QC Genie project drives this feature. 25% of incoming content receives replacement masters. Current process requires full re-QC. Goal: target QC only on changed parts. Multi-pass analysis: visual → audio → Pegasus language ID. Target: zero false negatives, <5% false positives.

### User Flow

1. **Select reference video** — dropdown picker from Library (indexed videos only)
2. **Find similar videos** — video-to-video search using averaged segment embeddings as video fingerprint, cosine similarity against all other indexed videos
3. **Show matches** — ranked list with similarity %, color-coded (green ≥85%, yellow 60-85%, dim <60%)
4. **Select a match** — opens integrated diff + side-by-side view

### Integrated Diff + Side-by-Side View

Each segment row is expandable. Click a row to reveal an inline synced dual-player for that segment pair.

**Segment row columns:**
- Row number
- Reference time range
- Status indicator (═══ match, ✕ missing, + added, ~ diff)
- Compare time range
- Similarity score %

**Color coding:**
- Default/matched: neutral
- Missing (in reference, not in compare): red `#ef4444` background tint
- Added (in compare, not in reference): green `#22c55e` background tint
- Changed (similar but divergent): yellow `#FABA17` background tint

**Expanded row (inline player):**
- Two video players side by side (REF label left, CMP label right)
- Shared playback controls (play/pause, prev/next segment)
- Synced playback indicator
- Per-modality similarity breakdown (visual/audio/transcription with colored dots)

**Summary stats bar** above segment rows:
- Matched count
- Missing count (red)
- Added count (green)
- Changed count (yellow)

### Video Fingerprint Computation

Each indexed video gets a precomputed fingerprint: one 512-d vector per modality, computed as the mean of all segment embeddings for that modality.

- **Storage:** New `video_fingerprints` collection in MongoDB (and/or S3 metadata object per video)
- **Schema:** `{ video_id, visual_fingerprint: [512], audio_fingerprint: [512], transcription_fingerprint: [512], segment_count, total_duration, created_at }`
- **Computation:** On video indexing (Lambda dual-writes fingerprint alongside segment embeddings). For existing videos, a backfill script computes from stored segments.
- **Find-similar query:** Compute cosine similarity of each modality fingerprint against all other videos. Overall similarity = weighted average (visual 0.5, audio 0.25, transcription 0.25). Return ranked results.

### Segment Alignment Algorithm

- Use cosine similarity between segment embeddings (not timestamp-based)
- Tolerance for time shifts (WBD noted 5-frame sensitivity issues — pure embedding match handles this naturally since matching is content-based, not time-based)
- **One-to-one greedy alignment:** Sort all cross-video segment pairs by similarity descending. Assign highest-similarity pair first, remove both segments from pool, repeat. This handles different segment counts naturally — unmatched segments become missing/added.
- Above 0.9 similarity: classify as **matched**
- Between 0.7-0.9: classify as **changed**
- Below 0.7 (or unmatched): classify as **missing** (reference-only) or **added** (compare-only)
- Matching uses the **visual modality embedding** as primary (most reliable for content identity), with audio/transcription scores computed post-alignment for the report breakdown

### Language Variant Auto-Detection

**Detection rule:** Visual similarity ≥90% AND (Audio <50% OR Transcription <50%) → flag as language variant.

When detected:
- Purple banner: "Language Variant Detected" with explanation
- Per-modality similarity bars showing the divergence pattern
- Segment alignment uses visual-only matching for these pairs

### Report

Accessed via "Report" button in the compare header. In-app scrollable view with export options.

**Report sections:**
1. **Summary** — reference name, compared name, overall similarity, durations, segment counts
2. **Segment breakdown** — matched/missing/added/changed counts with color-coded cards
3. **Modality similarity** — visual/audio/transcription similarity bars
4. **Language variant flag** — if detected
5. **Detailed segment table** — full row-by-row data

**Export options:**
- Export CSV — raw data (segment pairs, scores, statuses)
- Export PDF — formatted visual report

### API Endpoints (new)

- `POST /api/compare/find-similar` — find similar videos to a reference
  - Input: `{ video_id: string }`
  - Output:
    ```json
    {
      "reference": { "video_id": "abc", "name": "file.mp4", "segment_count": 12, "duration": 272.0 },
      "results": [
        {
          "video_id": "def",
          "name": "file_v2.mp4",
          "video_url": "https://cdn.../proxies/file_v2.mp4",
          "segment_count": 10,
          "duration": 238.0,
          "overall_similarity": 0.94,
          "modality_scores": { "visual": 0.96, "audio": 0.91, "transcription": 0.88 }
        }
      ]
    }
    ```

- `POST /api/compare/diff` — segment-level diff between two videos
  - Input: `{ reference_video_id: string, compare_video_id: string }`
  - Output:
    ```json
    {
      "summary": { "matched": 7, "missing": 4, "added": 1, "changed": 1, "overall_similarity": 0.78 },
      "language_variant": { "detected": false },
      "modality_similarity": { "visual": 0.82, "audio": 0.91, "transcription": 0.76 },
      "segments": [
        {
          "status": "matched",
          "similarity": 0.98,
          "reference": { "segment_id": 1, "start_time": 0.0, "end_time": 24.0 },
          "compare": { "segment_id": 1, "start_time": 0.0, "end_time": 24.0 },
          "modality_scores": { "visual": 0.99, "audio": 0.97, "transcription": 0.96 }
        },
        {
          "status": "missing",
          "similarity": null,
          "reference": { "segment_id": 3, "start_time": 48.0, "end_time": 72.0 },
          "compare": null,
          "modality_scores": null
        }
      ]
    }
    ```
    When `language_variant.detected` is true, includes: `{ "detected": true, "visual_similarity": 0.97, "audio_similarity": 0.31, "transcription_similarity": 0.12 }`

- `GET /api/compare/report/:reference_id/:compare_id` — generate report data
  - Calls diff internally, caches result for CSV/PDF export. Returns same structure as diff plus reference/compare video metadata.
- `GET /api/compare/report/:reference_id/:compare_id/csv` — export CSV
- `GET /api/compare/report/:reference_id/:compare_id/pdf` — export PDF (client-side generation using jsPDF from the in-app report HTML)

## Upload Page

### User Flow

1. **Drag & drop or browse** — drop zone accepts MP4, MOV, AVI, MKV
2. **Configure processing settings** — 2-column grid of options
3. **Click "Upload & Process"** — uploads file to S3, triggers processing
4. **Progress tracking** — per-file progress bar with status, settings summary, queue for multiple files

### Processing Settings

| Setting | Type | Options | Default |
|---------|------|---------|---------|
| Storage Backend | Multi-select | MongoDB, S3 Vectors | MongoDB |
| Index Mode | Multi-select | Single, Multi | Single |
| Segmentation | Single-select toggle | Dynamic, Fixed | Dynamic |
| Duration | Slider | 1-5s (dynamic), 1-30s (fixed) | 4s |
| Embedding Types | Multi-select toggles | Visual, Audio, Transcription | All three |

**Multi-select behavior:** Storage Backend and Index Mode use the same toggle pattern — select one or both. No explicit "Both" button.

### Upload Flow (backend)

1. Browser uploads video file via `POST /api/upload` (multipart, max 2GB)
2. FastAPI streams file to S3 input bucket (chunked upload via boto3 `upload_fileobj`, no full file in memory)
3. FastAPI invokes Lambda asynchronously with processing settings
4. Frontend polls `GET /api/upload/:upload_id/status` every 5 seconds for progress
5. Lambda progress is tracked via S3 marker objects (e.g., `status/{upload_id}.json`) written by Lambda at each stage
6. On completion, video appears in Library

### API Endpoints (new)

- `POST /api/upload` — multipart file upload
  - Input: video file + settings JSON (backend, index_mode, segmentation, duration, embedding_types)
  - Output: `{ upload_id: string, status: "processing" }`
- `GET /api/upload/:upload_id/status` — poll processing progress
  - Output: `{ status: "uploading" | "processing" | "complete" | "error", progress: number, message: string }`

## Technical Notes

- Video fingerprints are precomputed per-modality means of segment embeddings, stored in `video_fingerprints` collection
- Segment alignment uses visual embedding cosine similarity (content-based, handles time shifts naturally)
- Language variant detection is a heuristic (visual ≥90%, audio/transcription <50%)
- Upload streams through FastAPI to S3 (chunked, no full file buffered). Max file size: 2GB
- Progress tracking via polling (GET every 5s). Lambda writes status to S3 marker objects.
- PDF export is client-side (jsPDF) — no server-side PDF libraries needed
- All new pages are added to the existing single-page `static/index.html` architecture
- Existing `data-page` routing pattern extended for new pages
- Video-level similarity color coding (green ≥85%, yellow 60-85%, dim <60%) is distinct from segment-level thresholds (matched >0.9, changed 0.7-0.9, missing/added <0.7)
- Duration slider defaults: 4s for dynamic (min_duration_sec), 6s for fixed (segment_length_sec), matching existing Lambda defaults. Slider range and default reset when toggling segmentation mode.

## Mockups

Visual mockups saved in `.superpowers/brainstorm/67093-1773186240/`:
- `nav-approach-b.html` — sidebar before/after with Strand icons
- `compare-page.html` — initial state, similar videos, segment diff
- `compare-integrated.html` — integrated diff + side-by-side + language variant
- `compare-full.html` — report tab with export
- `upload-page-v3.html` — upload page with settings + progress (final version)
