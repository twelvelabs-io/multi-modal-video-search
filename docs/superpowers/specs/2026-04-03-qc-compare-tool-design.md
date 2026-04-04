# QC Compare Tool — Design Spec

## Goal

Add a QC comparison pipeline to the existing FastAPI app that compares video files against a master reference at the segment level using the embeddings already stored in MongoDB. Output is a JSON report with SMPTE timecodes that a QC operator can import into their timeline tool to jump to flagged segments.

VMAF runs as a secondary pixel-level signal on flagged segments.

## What Already Exists

- **MongoDB** (`multi_modal_video_search` database): 15 videos indexed with 1s segments, 512-dim embeddings across visual/audio/transcription modalities
- **`compare_client.py`**: `align_segments()` does positional and greedy segment alignment with cosine similarity, classifies matched/changed/missing/added
- **`app.py`**: `/api/compare/diff`, `/api/compare/multi-diff` endpoints already use this
- **S3** (`multi-modal-video-search-app`): Original MXF/MP4 files in `originals/`, proxies in `proxies/`
- **ffmpeg** available on the server (both system and static-vmaf builds)

## Videos

**Reference:** PH_Test2_F-010 (video_id: `7af9350893d589e5`, 273 segments)

**12 comparison videos** (excluding Basketball-w_mp3 and istockphoto noise):

| Blind Name | Video ID | Segments | Test Set |
|---|---|---|---|
| PH_Test1_A | b6007adb069d92fc | 273 | Test 1 (codec) |
| PH_Test1_B | be58a82bc2809819 | 273 | Test 1 (codec) |
| PH_Test1_C | 0a429be7f5685bf4 | 273 | Test 1 (codec) |
| PH_Test1_E | 89af42d77f1df93c | 273 | Test 1 (codec) |
| PH_Test2_A-009 | 1d0b54a7c2ef344b | 239 | Test 2 (content) |
| PH_Test2_B-003 | 4aeaba956a86adc6 | 273 | Test 2 (content) |
| PH_Test2_C-001 | 56a9671290f7d1e3 | 273 | Test 2 (content) |
| PH_Test2_D-007 | f64b50e5eb77544b | 271 | Test 2 (content) |
| PH_Test2_E-006 | 33accfd143dda86d | 267 | Test 2 (content) |
| PH_Test2_G-004 | 1fdf791a39ec22cd | 209 | Test 2 (content) |
| PH_Test2_H-011 | 53cd165fda8a8961 | 273 | Test 2 (content) |
| PH_Test2_I-008 | 282402cbd58987cf | 273 | Test 2 (content) |

## Architecture

No new indexing. No new embedding API calls. The pipeline reads existing embeddings from MongoDB and runs comparison logic.

```
MongoDB (existing embeddings)
    │
    ├── get_segments_for_video(ref_id)
    ├── get_segments_for_video(cmp_id)
    │
    ▼
align_segments(ref, cmp)          ← existing compare_client.py
    │
    ▼
classify_differences()            ← NEW: threshold-based classification
    │
    ▼
run_vmaf_segments()               ← NEW: ffmpeg libvmaf on flagged segments only
    │
    ▼
generate_report()                 ← NEW: JSON with SMPTE timecodes
```

## New Code

### `src/qc_report.py`

Single new file with all QC logic:

**`classify_differences(alignment_result)`**
- Takes output from `align_segments()` (already has per-segment similarity, status, modality scores)
- Applies thresholds to classify each flagged segment:
  - `>= 0.95` → match (compression artifacts only)
  - `0.85–0.95` → minor diff (color/luma changes, text overlays)
  - `< 0.85` → major diff (shot changes, removed content, reordering)
- Detects patterns:
  - **Timecode offset**: all segments match with constant time_shift
  - **Time compression/extension**: segments match but different count (already detected by align_segments when segment counts differ)
  - **Shot reorder**: a segment matches a reference segment at a very different position (large time_shift outlier)
  - **Added/removed**: segments with status "added" or "missing" from align_segments
  - **Color/luma change**: similarity 0.85–0.95 with high visual similarity relative to other modalities
- Consolidates adjacent flagged segments into contiguous difference entries
- No knowledge of expected changes — pure signal-based detection

**`run_vmaf_segment(ref_s3_key, cmp_s3_key, ref_start, ref_end, cmp_start, cmp_end)`**
- Downloads segment clips from S3 originals (or proxies as fallback)
- Runs ffmpeg with libvmaf: `ffmpeg -i ref_clip -i cmp_clip -lavfi libvmaf=log_fmt=json -f null -`
- Returns per-frame VMAF scores averaged over the segment
- Only called for flagged segments (not every segment — too expensive)

**`seconds_to_smpte(seconds, fps=29.97, start_tc="01:00:00:00")`**
- Converts seconds offset to SMPTE drop-frame timecode string
- Accounts for 01:00:00:00 start offset and 29.97 drop-frame

**`smpte_to_seconds(tc_string, fps=29.97, start_tc="01:00:00:00")`**
- Inverse of above

**`generate_report(ref_id, cmp_id, alignment, classifications, vmaf_scores)`**
- Assembles the JSON report per the spec format
- Includes: report_metadata, summary, differences array with SMPTE timecodes and per-segment detail

**`validate_report(report, test_matrix_entry)`**
- Post-hoc only — compares detected differences against ground truth by timecode proximity (2s tolerance)
- Computes true positives, false negatives, false positives
- Appends validation section to report
- This is the ONLY place the test matrix is used

### `POST /api/qc/run` endpoint in `app.py`

```python
{
    "reference_video_id": "7af9350893d589e5",
    "compare_video_ids": [...],  # optional, defaults to all 12
    "match_threshold": 0.95,     # optional
    "minor_diff_threshold": 0.85, # optional
    "run_vmaf": true,            # optional, default true
    "validate": false            # optional, run against test matrix
}
```

Returns:
```python
{
    "reports": [...],            # one report per comparison video
    "summary": {                 # aggregate across all comparisons
        "total_comparisons": 12,
        "avg_similarity": 0.93,
        "total_differences": 42
    }
}
```

Processing is sequential (one video at a time) to avoid memory pressure from VMAF downloads.

### `test_matrix.json`

Static file with ground truth from the spreadsheet. Only loaded when `validate: true` is passed. Structure:

```json
{
    "PH_Test2_D": {
        "test_id": "PH_2-1",
        "video_id": "f64b50e5eb77544b",
        "modifications": [
            {"type": "added_shot", "description": "River Monsters", "tc_in": "01:00:37:16", "tc_out": "01:00:44:14"},
            {"type": "removed_shot", "description": "Forest, climbing down rock", "tc_in": "01:01:58:24", "tc_out": "01:02:07:27"},
            {"type": "time_compressed", "description": "car", "tc": "01:03:46:28", "amount": "2%"},
            {"type": "time_extended", "tc": "01:04:00:25", "amount": "2%"}
        ]
    }
}
```

## Thresholds

Starting values — will calibrate with Test 1 results:

- `MATCH_THRESHOLD = 0.95` — above = identical
- `MINOR_DIFF_THRESHOLD = 0.85` — 0.85–0.95 = minor (color, text)
- Below 0.85 = major (shot changes, removed content)

Test 1 files (same content, different codec) should all score above 0.95. If not, lower the threshold.

## VMAF Notes

- Only run on segments flagged as minor or major diff (not every segment)
- Use originals from S3 when available (better quality for VMAF accuracy), fall back to proxies
- ffmpeg-vmaf static binary already on the server at `/usr/local/bin/ffmpeg-vmaf`
- Segment extraction: ffmpeg -ss {start} -t {duration} -i {input} to temp files
- VMAF comparison requires same resolution — scale both to 1920x1080 if needed

## Output Format

Matches the spec: JSON report with report_metadata, summary, differences array. Each difference has SMPTE timecodes, cosine similarity, VMAF score, severity, and type classification.
