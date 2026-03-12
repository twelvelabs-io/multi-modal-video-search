# Compare Page — 3D Cluster View Design Spec

## Goal

Replace the flat checkbox grid on the compare page with an interactive 3D cluster visualization where videos are automatically grouped by embedding similarity after indexing. Users navigate the space, search semantically, and select videos from clusters for comparison.

## Context

The compare page currently uses a flat grid of video cards with checkbox multi-select (implemented in the recent redesign). While functional, the user wants a spatial visualization that:
- Groups similar videos together visually (clusters)
- Lets you search semantically and fly to matching clusters
- Provides spatial navigation (pan, zoom) to explore the video library
- Preserves the click-to-select → NLE timeline comparison flow

The NLE timeline, video preview strip, zoom controls, segment legend, and detail panel from the existing redesign are **preserved**. This spec only replaces the video selection UI (the grid) with the 3D cluster view.

---

## 1. Cluster Generation

### Data Source

After indexing, the backend already stores per-video embeddings (visual, audio, transcription) in MongoDB/S3 Vectors. Clusters are computed from these embeddings.

### Clustering Algorithm

- **Method**: Agglomerative clustering on the combined (fused) embedding vectors
- **Distance metric**: Cosine distance
- **Threshold**: Distance threshold of 0.3 (videos within 0.3 cosine distance are grouped together)
- **Singletons**: Videos that don't cluster with any other video become single-item clusters
- **Computation**: Done server-side on the `/api/indexes/{backend}/{indexMode}/videos` response. New field `clusters` added to the response.

### API Response Shape

```json
{
  "videos": [ ...existing video objects... ],
  "clusters": [
    {
      "id": "cluster_0",
      "name": "Auto-generated label",
      "video_ids": ["vid1", "vid2", "vid3"],
      "centroid": [0.12, -0.34, ...],
      "avg_similarity": 0.84,
      "position": { "x": 0.22, "y": 0.25 }
    }
  ]
}
```

- `name`: Auto-generated from the most common words in video names within the cluster, or "Cluster N" fallback
- `position`: 2D projection of the cluster centroid using t-SNE or UMAP, normalized to 0-1 range
- `avg_similarity`: Mean pairwise cosine similarity between videos in the cluster

---

## 2. 3D Cluster View — Layout

### Viewport

The cluster view replaces the video grid area. Full width of the left panel (everything left of the selection side panel).

### Scene Elements

- **Grid floor**: Perspective-transformed grid lines providing depth cues. CSS `rotateX(70deg) rotateZ(15deg)` on a repeating linear-gradient background.
- **Cluster bubbles**: Circular elements positioned by their 2D projected coordinates (`position.x`, `position.y` from API). Size proportional to video count: `size = 60 + videoCount * 30` pixels (min 60px for singletons, capped at 220px).
- **Video dots**: Small colored circles inside each cluster bubble representing individual videos. Colors cycle through the track palette (`#00DC82`, `#6CD5FD`, `#FABA17`, `#FF6B6B`, `#A78BFA`, `#34D399`, `#F472B6`).
- **Similarity lines**: Dashed SVG lines between clusters. Opacity proportional to inter-cluster similarity. Only drawn between clusters with >0.5 similarity.
- **Cluster labels**: Below each bubble — cluster name + "avg N% similar" subtitle.
- **Axis labels**: "Visual similarity" on X axis, "Audio similarity" on Y axis. These are approximate — the 2D projection doesn't strictly map to single modalities, but gives the user spatial intuition.
- **Float animation**: Gentle sinusoidal drift (amplitude ~4-8px, period ~10-20s per cluster, phase-offset per cluster). Purely decorative.

### Fixed UI Overlays (Don't Move With Scene)

- **Header**: "Video Clusters" title + backend pills (MongoDB / Multi-Index)
- **Search bar**: Below header (see Section 4)
- **Zoom controls**: Bottom-right — minus, indicator (e.g., "1.2x"), plus, reset button
- **Nav hint**: Bottom-left — "drag → pan / scroll → zoom / click → select"

---

## 3. Spatial Navigation

### Pan

- **Drag** anywhere on the viewport (not on a cluster) to pan the entire scene
- Cursor changes to `grab` / `grabbing`
- Pan state: `panX`, `panY` in pixels, applied as `translate(panX, panY)` on the scene wrapper

### Zoom

- **Scroll wheel**: Multiplicative zoom (×0.9 or ×1.1 per tick) toward cursor position
- **+/− buttons**: ×1.2 per click
- **Reset button**: Returns to zoom=1.0, panX=0, panY=0
- **Range**: 0.3x to 4.0x
- Zoom toward cursor: adjusts pan so the point under the cursor stays fixed

### Camera Fly-To

Animated transitions when search results or chip clicks move the camera:
- `transition: transform 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94)` on the scene wrapper
- Single target: zoom to 1.8x, center on cluster
- Multiple targets: compute bounding box, zoom to fit with 15% padding

---

## 4. Semantic Search

### Search Bar

- Fixed overlay below the header, 320px wide
- Text input with magnifying glass icon + "Find" button
- Enter key triggers search, Escape clears

### Search Behavior

1. User types a query (e.g., "interview", "nature docs", "test encoding")
2. The query is embedded using the same Bedrock Marengo text embedding endpoint used for video search
3. Each cluster's centroid is compared to the query embedding via cosine similarity
4. Results:
   - **Strong match** (similarity ≥ 0.7): Green glow + `search-hit` class
   - **Partial match** (similarity ≥ 0.4): Amber glow + `search-partial` class
   - **No match** (< 0.4): Dimmed to 20% opacity + grayscale
5. Camera flies to fit all matching clusters
6. Result chips appear below search bar — one per matching cluster, sorted by score
   - Click a chip → fly to that cluster and select it

### Clear

- "clear" button in chip strip, or Escape key
- Removes all search highlights, resets camera to default position

### Edge Cases

- Empty query: no-op
- No matches: Show "No matching clusters" text, no dimming
- All match: Highlight all, camera stays at default zoom

---

## 5. Selection Panel

### Trigger

Click a cluster bubble → it gets `.selected` class (green accent border + glow). The right-side panel populates with that cluster's videos.

### Panel Layout

- **Header**: Cluster name + "N videos · avg X% similarity"
- **Video list**: Scrollable list of video cards, each with:
  - Checkbox (toggle-style square)
  - Thumbnail (48×30px, from `/api/thumbnail/{video_id}/1`)
  - Video name (first checked gets REF badge)
  - Codec/resolution tag
  - Similarity score to cluster centroid
  - MXF warning if applicable
- **Footer**: "Compare (N)" button → triggers `showTimelineDiff()` with checked videos

### Selection Logic

Same as the existing compare page redesign spec:
- First checked = REF (green badge)
- Subsequent = compare candidates
- Cap at 8 compare videos (9 total)
- ≥2 checked enables Compare button
- Unchecking REF promotes next checked video

### Cross-Cluster Selection

Users can select videos from different clusters:
- Click Cluster A → check some videos
- Click Cluster B → previous checks are preserved
- The panel shows the currently focused cluster's videos, but the Compare button count reflects ALL checked videos across all clusters
- A small "N selected from other clusters" indicator appears above the video list when cross-cluster selections exist

---

## 6. Visual Design

### Colors (TwelveLabs Design System)

- Background: `#1D1C1B`
- Surface: `#232221`, `#2A2928`
- Border: `#45423F`
- Text primary: `#F4F3F3`
- Text secondary: `#9B9895`
- Accent: `#00DC82`
- Accent muted: `rgba(0,220,130,0.12)`
- Warning: `#FABA17`
- Error: `#E22622`
- Track colors: `#00DC82`, `#6CD5FD`, `#FABA17`, `#FF6B6B`, `#A78BFA`, `#34D399`, `#F472B6`

### Fonts

- UI: Noto Sans
- Monospace (scores, zoom, technical metadata): IBM Plex Mono

### Button Style

Toggle-style (not solid green):
```css
background: var(--accent-muted);
border: 1px solid var(--accent);
color: var(--accent);
```

---

## 7. Integration With Existing Compare Page

### What Changes

- The video grid (`loadCompareVideos()` function, `.compare-grid` container) is replaced by the cluster view
- `compareCheckedIds` set is preserved — cluster view writes to it
- `compareCheckedVideos()` function is preserved — called by Compare button

### What Stays

- NLE timeline (tracks, ruler, playhead, consensus heatmap)
- Video preview strip (200px cards with mini players)
- Timeline zoom controls
- Segment legend
- Detail panel
- Stats bar
- Threshold slider + modality pills
- All CSS/JS for the timeline section

### Flow

1. Page loads → fetch videos + clusters from API
2. Render 3D cluster view
3. User navigates, searches, clicks cluster → panel shows videos
4. User checks videos across clusters → Compare (N) enables
5. Click Compare → `showTimelineDiff()` with selected video IDs
6. Timeline renders below (existing behavior)

---

## Scope Boundaries

**In scope:**
- 3D cluster view replacing the video selection grid
- Server-side clustering (agglomerative, cosine distance)
- 2D projection (t-SNE/UMAP) for cluster positioning
- Spatial navigation (pan, zoom, fly-to)
- Semantic search (query embedding → cluster centroid similarity)
- Search result highlighting (green/amber/dimmed) + chips
- Cross-cluster video selection
- Selection panel with video list

**Out of scope:**
- True 3D rendering (WebGL/Three.js) — CSS transforms are sufficient for this visualization
- Real-time re-clustering as videos are added
- Drag clusters to rearrange
- Cluster editing (rename, merge, split)
- Continuous video playback in cluster view
- Changes to the NLE timeline or video preview strip

**Backend changes required:**
- New clustering logic in the videos endpoint response
- 2D projection computation (t-SNE or UMAP on cluster centroids)
- Query embedding for semantic search (reuses existing Bedrock client)
