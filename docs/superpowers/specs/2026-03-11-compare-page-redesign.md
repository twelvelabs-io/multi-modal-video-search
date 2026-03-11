# Compare Page Redesign — Design Spec

## Goal

Fix the compare page's broken UX: replace the confusing 2-step video selection with checkbox multi-select, add video preview players to the NLE timeline, add track zoom, fix non-working controls, and replace solid green buttons with toggle-style pills.

## Context

The current compare page has several issues:
- Video selection requires a confusing 2-step flow (select ref → Find Similar → select candidates → Compare Selected)
- NLE timeline selector strip shows text-only video IDs, no thumbnails or player
- No zoom on timeline tracks
- Threshold slider and modality pills don't visually update segments (rendering bug)
- View filter pills (All/Diffs Only/Shifts) have no explanation
- Action buttons use solid green fill that clashes with the rest of the design system
- MXF files stored without transcoding show as unplayable black thumbnails

User approved: **Checkbox multi-select** for video selection, **Video preview cards with mini players** for the timeline strip.

---

## 1. Video Selection — Checkbox Multi-Select

**Replaces**: The current "Select Reference → Find Similar → Select candidates → Compare Selected" flow.

### Data Source

Videos are fetched from the existing `/api/indexes/{backend}/{indexMode}/videos` endpoint (same as current implementation). Thumbnails come from `/api/thumbnail/{video_id}/1` (first segment). Codec/resolution comes from `technical_metadata.video` on each video object.

### Layout

All indexed videos displayed in a responsive grid (`grid-template-columns: repeat(auto-fill, minmax(160px, 1fr))`). Each card has:

- **Thumbnail** (from existing thumbnail endpoint, or black + warning for unplayable files)
- **Checkbox** (top-left corner, styled as the toggle-check square from the upload page)
- **Video name** (truncated with ellipsis)
- **Codec/resolution tag** (from `technical_metadata`, e.g., "h264 · 1080p")
- **REF badge** — the first checked video automatically gets a green "REF" badge. Unchecking it promotes the next checked video to REF.
- **Unplayable indicator** — MXF files show a red warning tag ("MXF · no proxy") and can still be checked but with a tooltip warning that playback won't work.

### Selection Logic

- Click card → toggle checkbox (check/uncheck)
- First checked video = Reference (green accent border + REF badge)
- Subsequent checked videos = Compare candidates (default border)
- Cap at 8 compare videos (9 total including REF). Beyond that, clicking shows a toast "Maximum 8 comparison videos."
- When REF is unchecked: next video in **grid display order** (left-to-right, top-to-bottom) among remaining checked videos becomes REF.
- **0 videos selected**: Compare button shows "Compare" (disabled). No special message needed — the disabled state is sufficient.
- **1 video selected**: Compare button shows "Compare (1)" (disabled). The single video shows REF badge.
- **≥2 videos selected**: Compare button shows "Compare (N)" (enabled).

### Action Bar

Single button below the grid:
- **"Compare (N)"** — toggle-style button (muted green border, not solid fill). Enabled when ≥2 videos checked. Clicking triggers `showTimelineDiff()` with the first checked as `reference_id` and rest as `compare_ids`.
- Remove "Find Similar Videos" button from this flow entirely. The find-similar functionality stays on the search/analyze pages if needed.

### Button Style

All action buttons on the compare page switch from `.compare-find-btn` (solid green) to the toggle-style:
```css
/* Active/primary action — NOT solid green */
.compare-action-btn {
    padding: 6px 16px;
    border-radius: 6px;
    background: var(--accent-muted);   /* rgba(0,220,130,0.12) */
    border: 1px solid var(--accent);
    color: var(--accent);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
}
.compare-action-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}
```

This matches the MongoDB/Multi-Index toggle style the user prefers. Apply to: Compare button, Report button, Export JSON button.

---

## 2. NLE Timeline — Video Preview Strip

**Replaces**: The current text-only selector strip cards (116px wide, text-only). New cards are **200px wide** with thumbnail and player.

### Layout

Horizontal scrollable strip of **video preview cards** (200px wide each). Each card has:

- **Video thumbnail** (160×112 area) — uses the existing `/api/thumbnail/{video_id}/1` endpoint to get a poster frame. Falls back to a black card with play icon for missing thumbnails.
- **Mini progress bar** overlaid at bottom of thumbnail (thin 3px bar showing playhead position relative to duration)
- **Play/pause button** (small circular overlay, bottom-left of thumbnail). For unplayable files: grayed out with 50% opacity, cursor `not-allowed`, tooltip "File format not playable in browser."
- **Duration label** (bottom-right of thumbnail, IBM Plex Mono, e.g., "4:33")
- **Video name** (below thumbnail, truncated)
- **Codec/resolution tag** (IBM Plex Mono, e.g., "h264 · 1080p")
- **Color-coded border** — REF card gets green (`--accent`) border, compare cards get their track dot color (`#6CD5FD`, `#FABA17`, etc.)
- **Badge** — "REF" badge on reference card, position number (2, 3, 4...) on compare cards

### Interaction

- Click card body → toggle inclusion in comparison (same as current, but visually richer)
- Click play button → play/pause that specific video (HTML5 `<video>` element). Each card plays independently — there is no continuous sync between players.
- Playhead drag on timeline → seek all video players to that time position (one-shot seek, not continuous sync). This is a seek operation, not playback synchronization.

### Video Player

Each card contains an actual `<video>` element (hidden until play is clicked, thumbnail serves as poster). Source URL comes from the `video_urls` dict in the multi-diff API response (available after comparison is triggered).

### Edge Case: All Videos Unplayable

If all selected videos are MXF/unplayable, the preview strip still renders with thumbnails (from the thumbnail API endpoint which works regardless of video format — it extracts frames server-side). Play buttons are all disabled. The comparison and timeline still work normally since they're embedding-based.

---

## 3. Timeline — Zoom Controls

**New feature**: Zoom slider in the controls bar, right side.

### Controls

- **"−" button** — zoom out (decrease zoom level)
- **Zoom slider** — range input, min 0.5x to max 5x, default 1x
- **"+" button** — zoom in (increase zoom level)
- **Zoom level label** — displays current level (e.g., "1.5x") in IBM Plex Mono

### Implementation

- `tlState.zoom = 1.0` — new state field
- Zoom multiplies the timeline track bar width: `trackBar.style.width = (100 * zoom) + '%'`
- The timeline container (`#tlTimeline`) gets `overflow-x: auto` to allow horizontal scrolling when zoomed
- Ruler ticks recalculate on zoom change
- Playhead position recalculates on zoom change, accounting for `scrollLeft` offset
- Alt+scroll wheel on timeline area → zoom in/out (Alt used to avoid conflict with browser Ctrl+scroll zoom)
- Zoom is client-side only, no API re-call

---

## 4. Controls Bar — Fix Non-Working Controls

The threshold slider and modality pills have working JavaScript handlers but the visual feedback isn't updating properly. These are investigation/debug items — the implementer should trace the rendering pipeline to find the root cause.

**Note**: `tlState.threshold` is stored as an **integer 0-100** (matching the existing codebase), NOT as a float 0-1. The original NLE design spec showed `0.85` but the implementation correctly uses `85`. All threshold comparisons use `score < tlState.threshold / 100`.

### Threshold Slider

The slider calls `renderTimeline()` which rebuilds tracks. The `segColorClass()` function checks `score < tlState.threshold / 100` to add the `alert` class. Investigate: verify `getSegScore()` returns the right value based on `tlState.modalityMode` and that the rebuilt DOM actually reflects the new classes.

### Modality Pills

`setTimelineModality()` sets `tlState.modalityMode` and calls `renderTimeline()`. Investigate: verify `getSegScore()` correctly reads `seg.modality_scores[tlState.modalityMode]` when mode is not "combined", and that segment color classes change visually.

### Pill Style Change

Active pills switch from solid green fill to toggle-style:
```css
.tl-pill.active {
    background: var(--accent-muted);  /* was: var(--accent) */
    color: var(--accent);              /* was: var(--text-inverse) */
    border-color: var(--accent);
}
```

---

## 5. View Filters — Explanation

Add a **segment legend** below the timeline (always visible):

```
[green block] Matched (≥90%)  [yellow block] Changed (70-90%)  [red block] Missing  [yellow-left-border block] Time-shifted  [pulsing-red-outline block] Below threshold (alert overlay, can appear on any status)
```

The "Below threshold" entry in the legend is a visual modifier (the `alert` class), not a segment status. It can appear on top of matched, changed, or any other status when similarity is below the slider value.

The filter pills (All / Diffs Only / Shifts) get tooltips on hover:
- **All**: "Show every segment on every track"
- **Diffs Only**: "Hide matched segments, show only changed/missing/added"
- **Shifts**: "Show only segments offset by >0.5s"

---

## 6. MXF / Unplayable File Handling

**Root cause**: PH_Test1_C.mxf was uploaded but Lambda's transcode step didn't convert it to MP4. The MXF file was stored directly to `proxies/`.

**Frontend handling** (this spec):
- Detect unplayable files by checking the file extension from `s3_key` or `metadata.s3_key`
- Known unplayable extensions: `.mxf`, `.r3d`, `.braw`, `.ari`
- Note: `.mov` files are NOT blanket-flagged. Most `.mov` files (H.264/AAC) are browser-playable. Only ProRes-encoded `.mov` would fail, but detecting codec requires backend metadata — handle this with a player error fallback (see below) rather than preemptive blocking.
- Show warning badge on card: red text, e.g., "MXF · no proxy"
- Disable play button on the preview card (grayed out, tooltip)
- **Player error fallback**: If a `.mov` or any video fails to play (`<video>` `error` event), show an overlay on the thumbnail: "Format not supported" and disable the play button retroactively.
- Video can still be compared (embedding-based comparison works regardless of playback)

**Backend fix** (separate scope — Lambda transcode pipeline): Out of scope for this spec. The MediaConvert plan already covers this.

---

## Preserved Components (Unchanged)

The following NLE timeline components from the original design spec are preserved as-is:

- **Consensus heatmap** — single stacked bar below tracks using `Math.min` of comparison similarities
- **Detail panel** — segment header, frame grid, modality breakdown, "Analyze with AI" button, "Export Segment" button
- **Stats bar** — Total | Matched | Changed | Missing | Below threshold | LANG badge
- **Playhead** — draggable vertical line with time label, snaps to segment boundaries

---

## Scope Boundaries

**In scope:**
- Checkbox multi-select grid replacing 2-step selection flow
- Video preview cards with mini players in timeline strip (200px cards, up from 116px)
- Timeline zoom controls (0.5x–5x with Alt+scroll)
- Toggle-style buttons/pills replacing solid green
- Segment legend with tooltip explanations
- MXF/unplayable file indicators with player error fallback
- Debug threshold/modality rendering bugs
- Pill style change from solid to toggle

**Out of scope:**
- Lambda transcode pipeline fix for MXF files (covered by MediaConvert plan)
- Continuous video playback sync across players (one-shot seek on playhead drag IS in scope)
- Find Similar functionality (preserved on other pages, removed from compare page)
- Waveform visualization
- Drag-to-select time ranges

**Backward compatibility:**
- The `/api/compare/multi-diff` endpoint is unchanged
- The `tlState` object gets one new field (`zoom`). Existing fields (`selectedCompareIdx`, `refDuration`, etc.) are preserved.
- Existing NLE timeline CSS/JS is modified in-place, not replaced
