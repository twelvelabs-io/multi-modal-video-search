# NLE Timeline Compare View — Design Spec

## Goal

Replace the current table-based compare diff view with a professional NLE (Non-Linear Editor) timeline interface that supports N-video simultaneous comparison, time-shift detection, threshold-based highlighting, and a draggable playhead.

## Context

The current compare view uses a flat segment table with expandable rows. It works for 2-video comparison but doesn't scale visually to N videos, offers no timeline perspective, and can't surface time shifts between corresponding segments. Users need a broadcast QC / localization QC tool that immediately reveals where videos diverge.

## Design Decision: Option A — Stacked NLE Tracks

User approved a Premiere/Resolve-style stacked timeline with consensus heatmap after evaluating three options (Stacked NLE Tracks, Pairwise Heatmap Matrix, Swimlane + Waveform). The final mockup uses the TwelveLabs design system.

---

## Architecture

### Layout Structure

```
+--------+-----------------------------------------------------------+
| Sidebar|  Top Bar: "Compare Videos"   [Report] [Export JSON]       |
|        +---------------------------------------------------------+
|  nav   |  Video Selector Strip (horizontal scroll, card carousel) |
|        +---------------------------------------------------------+
|        |  Controls Bar: [Threshold ▬▬▬] [Combined|Vis|Aud|Spch]  |
|        |                                        [All|Diffs|Shifts]|
|        +---------------------------------------------------------+
|        |  Timeline Ruler: 0:00  0:30  1:00  1:30  2:00 ...       |
|        |  ┌─ REF ──────────────────────────────────────────┐      |
|        |  │ ████ ████ ████ ████ ████ ████ ████ ████ ████  │      |
|        |  ├─ Video 2 ─────────────────────────────────────┤      |
|        |  │ ████ ████ ▓▓▓▓ ████ ████ ░░░░ ████ ████ ████ │      |
|        |  ├─ Video 3 ─────────────────────────────────────┤      |
|        |  │ ████ ████ ████ ▓▓▓▓ ████ ████ ████ ░░░░ ████ │      |
|        |  └───────────────────────────────────────────────┘      |
|        |  Consensus Heatmap (stacked color bar)                   |
|        |  Playhead: │ 1:06.3                                      |
|        +---------------------------------------------------------+
|        |  Detail Panel (expandable): Segment 67 — 1:06 → 1:07    |
|        |  [Frame grid] [Modality breakdown] [Analyze AI] [Export] |
|        +---------------------------------------------------------+
|        |  Stats Bar: 273 total | 258 matched | 9 changed | 6 miss|
+--------+-----------------------------------------------------------+
```

### Components

#### 1. Video Selector Strip
- Horizontal scrollable card carousel (116px per card)
- First card = REF (green accent border + "REF" badge)
- Other cards show position number (2, 3, 4...)
- Click to toggle inclusion in comparison
- Shows thumbnail, filename truncated, codec/resolution tag

#### 2. Controls Bar
- **Threshold slider**: Range 0-100%, gradient fill (red → yellow → green). **Client-side only — does not re-invoke the API.** Segments with similarity below the slider value receive the `alert` visual treatment (pulsing red outline). The backend matching floor stays fixed at 0.7.
- **Modality pills**: Combined | Visual | Audio | Speech — toggle which modality drives the segment coloring. "Combined" uses weighted average; others use single-modality score
- **View filter pills**: All | Diffs Only | Shifts — filter which segments are visible on the timeline. "Diffs Only" hides matched segments. "Shifts" shows only segments where `abs(time_shift) > 0.5s`.

#### 3. Timeline Tracks
- One track per video, stacked vertically
- **Track header** (left label): video name, colored dot, filename, codec/resolution metadata
- **Track bar**: proportionally-sized segment blocks using `flex-basis` on duration ratio, with `min-width: 4px` to keep tiny segments clickable
- **Segment blocks** colored by status (`status` field from API):
  - `matched` (similarity >= 0.9): green-tinted (`--success` at 15% opacity)
  - `changed` (0.7 <= sim < 0.9): yellow-tinted (`--warning` at 15% opacity)
  - `missing` (ref segment not in compare): red-tinted (`--error` at 20% opacity), pulsing red outline
  - `added` (extra segment in compare): green-tinted with "+" marker
- **Visual modifiers** (applied on top of status, not separate status values):
  - `shifted`: yellow left-border accent, applied when `abs(time_shift) > 0.5s`
  - `alert`: pulsing red outline animation, applied when similarity < threshold slider value (client-side only)
- **Segment tooltip on hover**: "Seg N · X%" or "MISSING"
- **Click segment**: opens detail panel below

#### 4. Playhead
- Vertical white line with triangle head, spanning all tracks
- Displays current time in `IBM Plex Mono` font
- Draggable horizontally across the timeline
- Snaps to segment boundaries on release

#### 5. Consensus Heatmap
- Single stacked bar below all tracks
- Uses the **reference track's segment boundaries** as the grid. For each reference segment, take `Math.min(...comparisons.map(c => c.similarity || 0))` from the aligned segments
- Provides at-a-glance summary: solid green = all match, red spots = problems

#### 6. Detail Panel
- Expands when a segment (or playhead position) is selected
- Header: "Segment N — start → end"
- Status badges: count of videos below threshold, count shifted
- **Frame grid**: one card per video (REF + comparisons)
  - Thumbnail frame (extracted from S3)
  - Similarity score badge
  - Shift amount (e.g., "+0.3s")
- **Modality breakdown table**: Visual %, Audio %, Speech % per comparison video
- **Actions**: "Analyze with AI" (analyzes reference vs. the focused comparison card — click a card to select it), "Export Segment"

#### 7. Stats Bar (bottom)
- Total segments | Matched (green) | Changed (yellow) | Missing (red) | Below threshold (red)
- Language variant indicator if detected (purple "LANG" badge)

---

## Backend Changes

### 1. `align_segments()` — Time Shift Detection

Add `time_shift` field to each aligned segment:

```python
# For each matched/changed pair:
alignment["time_shift"] = round(cmp_start - ref_start, 2)  # seconds
```

Add shift summary to the response:

```python
"summary": {
    ...existing fields...,
    "shifted": count_of_segments_with_abs_shift_gt_0.5s,
    "avg_shift": mean_shift_seconds
}
```

### 2. `align_segments()` — Configurable Threshold

Accept `threshold` parameter (default 0.7) to replace the hard-coded cutoff. This controls the **greedy matching floor** — pairs below this similarity are classified as `missing`/`added` rather than `changed`. The `matched`/`changed` boundary at 0.9 remains fixed.

```python
def align_segments(ref_segments, cmp_segments, threshold=0.7):
```

Note: The UI threshold slider is separate and client-side only. It controls which segments get the `alert` visual treatment, without re-invoking the API.

### 3. Multi-Video Compare Endpoint

Currently the diff endpoint compares 2 videos. Extend to support N videos (recommended cap: 8 comparison videos).

**Endpoint**: `POST /api/compare/multi-diff`

**Request**:
```json
{
  "reference_id": "video_abc",
  "compare_ids": ["video_def", "video_ghi", "video_jkl"]
}
```

**Response** (full shape):
```json
{
  "reference": {
    "video_id": "video_abc",
    "metadata": { "name": "...", "duration": 273.5, "s3_key": "..." },
    "technical_metadata": { "video": {...}, "audio": {...}, "container": {...} }
  },
  "comparisons": [
    {
      "video_id": "video_def",
      "metadata": { "name": "...", "duration": 273.5, "s3_key": "..." },
      "technical_metadata": { "video": {...}, "audio": {...}, "container": {...} },
      "alignment": {
        "summary": {
          "matched": 258, "changed": 9, "missing": 6, "added": 0,
          "shifted": 3, "avg_shift": 0.4,
          "overall_similarity": 0.94
        },
        "language_variant": { "detected": false },
        "modality_similarity": { "visual": 0.95, "audio": 0.92, "transcription": 0.89 },
        "segments": [
          {
            "status": "matched",
            "similarity": 0.97,
            "time_shift": 0.0,
            "reference": { "segment_id": 1, "start_time": 0.0, "end_time": 1.0 },
            "compare": { "segment_id": 1, "start_time": 0.0, "end_time": 1.0 },
            "modality_scores": { "visual": 0.98, "audio": 0.95, "transcription": 0.97 }
          },
          {
            "status": "changed",
            "similarity": 0.82,
            "time_shift": 0.3,
            "reference": { "segment_id": 5, "start_time": 4.0, "end_time": 5.0 },
            "compare": { "segment_id": 6, "start_time": 4.3, "end_time": 5.3 },
            "modality_scores": { "visual": 0.85, "audio": 0.78, "transcription": 0.84 }
          },
          {
            "status": "missing",
            "similarity": null,
            "time_shift": null,
            "reference": { "segment_id": 10, "start_time": 9.0, "end_time": 10.0 },
            "compare": null,
            "modality_scores": null
          }
        ]
      }
    }
  ]
}
```

Implementation: Loop `align_segments()` for each compare video against the reference. Each alignment is independent. Use `asyncio.gather` to parallelize the N calls.

The existing `/api/compare/diff` endpoint is preserved for backward compatibility.

### 4. Consensus Computation (Frontend)

Computed client-side from the multi-diff response. For each reference segment position, take `Math.min(...comparisons.map(c => alignedSegmentSimilarity))`. No backend change needed.

---

## Frontend State

Since the app uses vanilla JS, the timeline manages state via a single state object:

```javascript
const timelineState = {
  threshold: 0.85,         // UI slider value (client-side alert cutoff)
  modalityMode: 'combined', // 'combined' | 'visual' | 'audio' | 'transcription'
  viewFilter: 'all',       // 'all' | 'diffs' | 'shifts'
  selectedSegmentIdx: null, // index into reference segments
  includedVideoIds: [],     // which videos are toggled on
  playheadTime: 0,         // current playhead position in seconds
  apiResponse: null         // cached multi-diff response
};
```

**Re-render strategy**: Changing `threshold`, `modalityMode`, or `viewFilter` triggers a DOM re-render of the timeline tracks and consensus bar from the cached `apiResponse` — no API re-call. Changing `includedVideoIds` triggers a new API call (if new videos added) or filters the cached response (if removing videos). `selectedSegmentIdx` toggles the detail panel.

**Loading state**: Show a skeleton shimmer on tracks while the multi-diff endpoint computes. If one comparison fails, show its track with an error banner and proceed with the rest (partial results).

**Empty state**: When no compare videos are selected, show a prompt: "Select videos above to compare against the reference."

---

## Design Tokens (TwelveLabs Design System)

All colors reference existing CSS custom properties from the app:

| Purpose | Variable | Value |
|---------|----------|-------|
| Body background | `--bg-body` | `#1D1C1B` |
| Surface/panels | `--bg-surface` | `#2A2928` |
| Elevated cards | `--bg-elevated` | `#333231` |
| Borders | `--border` | `#45423F` |
| Primary text | `--text-primary` | `#F4F3F3` |
| Secondary text | `--text-secondary` | `#9B9895` |
| Accent/selected | `--accent` | `#00DC82` |
| Matched segments | `--success` | `#60E21B` |
| Changed segments | `--warning` | `#FABA17` |
| Missing segments | `--error` | `#E22622` |
| Visual modality | `--visual` | `#6CD5FD` |
| Audio modality | `--audio` | `#60E21B` |
| Transcription | `--transcription` | `#FABA17` |
| Font UI | Noto Sans | — |
| Font mono/time | IBM Plex Mono | — |

---

## Interaction Spec

| Action | Result |
|--------|--------|
| Hover segment | Brightness +0.2, tooltip shows "Seg N · X%" |
| Click segment | Detail panel expands with frame grid + modality breakdown |
| Drag playhead | Playhead moves, time label updates, detail panel follows |
| Drag threshold slider | Segments below new threshold gain pulsing red outline |
| Toggle modality pill | Segment colors recalculate using selected modality score |
| Toggle view filter | Timeline shows/hides segments based on filter |
| Click "Analyze with AI" | Calls existing analyze-segment endpoint |
| Click video card | Toggles video inclusion in comparison |

---

## Scope Boundaries

**In scope:**
- NLE timeline with stacked tracks, colored segments, playhead, consensus bar
- N-video simultaneous comparison (1 ref + N compare)
- Time shift detection and display (per-segment shift field)
- Threshold slider with dynamic highlighting
- Modality toggle (Combined/Visual/Audio/Speech)
- View filters (All/Diffs Only/Shifts)
- Detail panel with frame grid and modality breakdown
- Multi-diff API endpoint
- Reuse existing `align_segments()` with extensions

**Backward compatibility**: The existing `/api/compare/diff` endpoint is preserved. The table-based segment view is replaced by the NLE timeline. Report and CSV export endpoints are unchanged.

**Out of scope:**
- Video playback sync (future enhancement)
- Waveform visualization
- Segment splitting/merging detection
- Weighted multi-modality matching in the greedy phase
- Temporal windowing constraints in alignment algorithm
- Drag-to-select time ranges
