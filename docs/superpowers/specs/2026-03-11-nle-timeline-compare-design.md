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
- **Threshold slider**: Range 0-100%, gradient fill (red → yellow → green). Controls the "alert" cutoff — segments below this threshold get a pulsing red outline
- **Modality pills**: Combined | Visual | Audio | Speech — toggle which modality drives the segment coloring. "Combined" uses weighted average; others use single-modality score
- **View filter pills**: All | Diffs Only | Shifts — filter which segments are visible on the timeline. "Diffs Only" hides matched segments, "Shifts" shows only time-shifted segments

#### 3. Timeline Tracks
- One track per video, stacked vertically
- **Track header** (left label): video name, colored dot, filename, codec/resolution metadata
- **Track bar**: proportionally-sized segment blocks using `flex-basis` on duration ratio
- **Segment blocks** colored by status:
  - `matched` (similarity >= 0.9): green-tinted (`--success` at 15% opacity)
  - `changed` (0.7 <= sim < 0.9): yellow-tinted (`--warning` at 15% opacity)
  - `missing` (ref segment not in compare): red-tinted (`--error` at 20% opacity), pulsing red outline
  - `added` (extra segment in compare): green-tinted with "+" marker
  - `shifted` (time offset detected): yellow left-border accent
  - `alert` (below threshold): pulsing red outline animation
- **Segment tooltip on hover**: "Seg N · X%" or "MISSING"
- **Click segment**: opens detail panel below

#### 4. Playhead
- Vertical white line with triangle head, spanning all tracks
- Displays current time in `IBM Plex Mono` font
- Draggable horizontally across the timeline
- Snaps to segment boundaries on release

#### 5. Consensus Heatmap
- Single stacked bar below all tracks
- Each position shows the "worst" status across all comparison videos at that time
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
- **Actions**: "Analyze with AI" (calls existing `/api/compare/analyze-segment`), "Export Segment"

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

Accept `threshold` parameter (default 0.7) to replace the hard-coded cutoff:

```python
def align_segments(ref_segments, cmp_segments, threshold=0.7):
```

### 3. Multi-Video Compare Endpoint

Currently the diff endpoint compares 2 videos. Extend to support N videos:

**Endpoint**: `POST /api/compare/multi-diff`

**Request**:
```json
{
  "reference_id": "video_abc",
  "compare_ids": ["video_def", "video_ghi", "video_jkl"],
  "threshold": 0.85
}
```

**Response**:
```json
{
  "reference": { "video_id": "...", "metadata": {...}, "technical_metadata": {...} },
  "comparisons": [
    {
      "video_id": "video_def",
      "metadata": {...},
      "technical_metadata": {...},
      "alignment": { ...existing align_segments output with time_shift... }
    },
    ...
  ]
}
```

Implementation: Loop `align_segments()` for each compare video against the reference. Each alignment is independent.

### 4. Consensus Computation (Frontend)

Computed client-side from the multi-diff response. For each time position, take the minimum similarity score across all comparisons. No backend change needed.

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

**Out of scope:**
- Video playback sync (future enhancement)
- Waveform visualization
- Segment splitting/merging detection
- Weighted multi-modality matching in the greedy phase
- Temporal windowing constraints in alignment algorithm
- Drag-to-select time ranges
