# Compare Page Redesign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the compare page UX — checkbox multi-select for videos, video preview strip with mini players, timeline zoom, toggle-style buttons, segment legend, and MXF handling.

**Architecture:** All changes in `static/index.html` (monolithic frontend). CSS changes first, then HTML structure, then JS logic. No backend changes needed — the `/api/compare/multi-diff` endpoint and `/api/indexes/{backend}/{mode}/videos` endpoint are already sufficient.

**Tech Stack:** Vanilla JS, CSS custom properties (TwelveLabs design system), HTML5 `<video>` element.

**Spec:** `docs/superpowers/specs/2026-03-11-compare-page-redesign.md`

---

## File Structure

- Modify: `static/index.html` — all CSS, HTML, and JS changes

No new files. The existing monolithic `index.html` contains all styles, markup, and scripts.

---

## Chunk 1: CSS + HTML + Button Style Changes

### Task 1: Replace Button Styles and Pill Active State

**Files:**
- Modify: `static/index.html:3299-3315` (`.compare-find-btn` CSS)
- Modify: `static/index.html:3791` (`.tl-pill.active` CSS)

- [ ] **Step 1: Replace `.compare-find-btn` with `.compare-action-btn`**

Find the existing `.compare-find-btn` block at lines 3299-3315 and replace it:

```css
/* Old — lines 3299-3315 */
.compare-find-btn {
    margin-top: 12px;
    padding: 10px 24px;
    background: var(--accent);
    color: var(--text-inverse);
    border: none;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
}
.compare-find-btn:hover { background: var(--accent-hover); }
.compare-find-btn:disabled { opacity: 0.4; cursor: not-allowed; }
```

Replace with:

```css
.compare-action-btn {
    margin-top: 12px;
    padding: 6px 16px;
    background: var(--accent-muted);
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
}
.compare-action-btn:hover { background: rgba(0,220,130,0.18); }
.compare-action-btn:disabled { opacity: 0.4; cursor: not-allowed; }
```

- [ ] **Step 2: Change `.tl-pill.active` from solid green to toggle style**

Find at line 3791:

```css
.tl-pill.active { background: var(--accent); color: var(--text-inverse); border-color: var(--accent); }
```

Replace with:

```css
.tl-pill.active { background: var(--accent-muted); color: var(--accent); border-color: var(--accent); }
```

- [ ] **Step 3: Add zoom control CSS**

Add after the `.tl-pills-right` rule (around line 3793):

```css
.tl-zoom-group {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-left: 8px;
}
.tl-zoom-btn {
    width: 22px;
    height: 22px;
    border-radius: 4px;
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
    padding: 0;
}
.tl-zoom-btn:hover { border-color: var(--border-hover); color: var(--text-primary); }
.tl-zoom-slider {
    width: 60px;
    height: 4px;
    -webkit-appearance: none;
    appearance: none;
    background: var(--border);
    border-radius: 2px;
    outline: none;
}
.tl-zoom-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--text-primary);
    border: 1.5px solid var(--border);
    cursor: pointer;
}
.tl-zoom-slider::-moz-range-thumb {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--text-primary);
    border: 1.5px solid var(--border);
    cursor: pointer;
}
.tl-zoom-val {
    font-size: 9px;
    color: var(--text-secondary);
    font-family: 'IBM Plex Mono', monospace;
    min-width: 24px;
}
```

- [ ] **Step 4: Add `tl-pulse` keyframe animation**

Add after zoom CSS (this is referenced by the legend's "Below threshold" swatch):

```css
@keyframes tl-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
```

- [ ] **Step 5: Add segment legend CSS**

Add after the keyframe:

```css
.tl-legend {
    display: flex;
    gap: 14px;
    font-size: 9px;
    color: var(--text-secondary);
    padding: 6px 4px;
    margin-top: 6px;
    flex-wrap: wrap;
}
.tl-legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
}
.tl-legend-swatch {
    width: 20px;
    height: 10px;
    border-radius: 1px;
    display: inline-block;
}
```

- [ ] **Step 6: Add checkbox card CSS for video selection grid**

Add after `.compare-video-card.selected` rules (around line 3290):

```css
.compare-video-card .card-checkbox {
    position: absolute;
    top: 6px;
    left: 6px;
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1.5px solid var(--border);
    background: rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    color: transparent;
    transition: all 0.15s;
    z-index: 2;
}
.compare-video-card.selected .card-checkbox {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(0,220,130,0.15);
}
.compare-video-card .card-ref-badge {
    position: absolute;
    top: 6px;
    right: 6px;
    font-size: 8px;
    font-weight: 700;
    background: var(--accent);
    color: var(--bg-body);
    padding: 1px 5px;
    border-radius: 2px;
    z-index: 2;
    display: none;
}
.compare-video-card.is-ref .card-ref-badge { display: block; }
.compare-video-card .card-warning {
    font-size: 8px;
    color: var(--error);
    margin-top: 2px;
}
.compare-video-card .card-codec {
    font-size: 8px;
    color: var(--text-tertiary);
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 1px;
}
```

- [ ] **Step 7: Update HTML — Replace buttons with `.compare-action-btn` class**

In the HTML at lines 4976-4978, change button classes:

```html
<!-- Old -->
<button id="compareReportBtn" class="compare-find-btn" ...>Report</button>
<button id="compareExportBtn" class="compare-find-btn" ...>Export JSON</button>
```

Replace with:

```html
<button id="compareReportBtn" class="compare-action-btn" ...>Report</button>
<button id="compareExportBtn" class="compare-action-btn" ...>Export JSON</button>
```

Also at lines 5003-5004, replace the Find Similar and Compare Selected buttons:

```html
<!-- Old -->
<button id="compareFindBtn" class="compare-find-btn" onclick="findSimilarVideos()" disabled>Find Similar Videos</button>
<button id="compareSelectedBtn" class="compare-find-btn" style="display:none;margin-left:8px" onclick="compareSelectedVideos()">Compare Selected (0)</button>
```

Replace with a single button:

```html
<button id="compareBtn" class="compare-action-btn" onclick="compareCheckedVideos()" disabled>Compare</button>
```

- [ ] **Step 8: Add zoom controls to HTML controls bar**

In the controls bar HTML (lines 5019-5036), add zoom group after the view pills div (after line 5035):

```html
<div class="tl-zoom-group">
    <button class="tl-zoom-btn" onclick="tlZoom(-0.5)">−</button>
    <input type="range" class="tl-zoom-slider" id="tlZoomSlider" min="50" max="500" value="100" step="25">
    <button class="tl-zoom-btn" onclick="tlZoom(0.5)">+</button>
    <span class="tl-zoom-val" id="tlZoomVal">1x</span>
</div>
```

- [ ] **Step 9: Add segment legend HTML below stats bar**

After the stats bar `<div class="tl-stats" id="tlStats"></div>` (line 5065), add:

```html
<div class="tl-legend" id="tlLegend">
    <div class="tl-legend-item"><span class="tl-legend-swatch" style="background:rgba(96,226,27,0.15)"></span> Matched (≥90%)</div>
    <div class="tl-legend-item"><span class="tl-legend-swatch" style="background:rgba(250,186,23,0.20)"></span> Changed (70-90%)</div>
    <div class="tl-legend-item"><span class="tl-legend-swatch" style="background:rgba(226,38,34,0.20);outline:1px solid var(--error)"></span> Missing</div>
    <div class="tl-legend-item"><span class="tl-legend-swatch" style="background:rgba(250,186,23,0.20);border-left:2px solid var(--warning)"></span> Time-shifted</div>
    <div class="tl-legend-item"><span class="tl-legend-swatch" style="background:rgba(226,38,34,0.20);outline:1px solid var(--error);animation:tl-pulse 1.5s infinite"></span> Below threshold (alert)</div>
</div>
```

- [ ] **Step 10: Add tooltips to view filter pills**

Update the view pill buttons in HTML (lines 5032-5035) to add `title` attributes:

```html
<button class="tl-pill active" data-view="all" onclick="setTimelineView(this)" title="Show every segment on every track">All</button>
<button class="tl-pill" data-view="diffs" onclick="setTimelineView(this)" title="Hide matched segments, show only changed/missing/added">Diffs Only</button>
<button class="tl-pill" data-view="shifts" onclick="setTimelineView(this)" title="Show only segments offset by >0.5s">Shifts</button>
```

- [ ] **Step 11: Verify and commit**

Open the app, navigate to Compare page. Verify:
- Buttons use toggle style (muted green border, not solid fill)
- Pills use toggle style when active
- Legend appears below stats bar
- View filter pills show tooltips on hover

```bash
git add static/index.html
git commit -m "style(compare): toggle-style buttons, pill active state, zoom CSS, legend, tooltips"
```

---

## Chunk 2: Video Selection — Checkbox Multi-Select

### Task 2: Rewrite Video Selection Grid with Checkboxes

**Files:**
- Modify: `static/index.html:8145` (`compareSelectedIds` declaration area)
- Modify: `static/index.html:8167-8221` (`loadCompareVideos` function)
- Modify: `static/index.html:8370-8373` (`compareSelectedVideos` function)

- [ ] **Step 1: Add `compareCheckedIds` array and helper**

At line 8145, after `let compareSelectedIds = [];`, add:

```javascript
let compareCheckedIds = [];  // ordered list of checked video IDs for new compare flow

function updateCompareBtn() {
    const btn = document.getElementById('compareBtn');
    if (!btn) return;
    btn.disabled = compareCheckedIds.length < 2;
    btn.textContent = compareCheckedIds.length > 0 ? `Compare (${compareCheckedIds.length})` : 'Compare';
}

function updateRefBadge() {
    const grid = document.getElementById('compareVideoGrid');
    if (!grid) return;
    grid.querySelectorAll('.compare-video-card').forEach(card => {
        card.classList.remove('is-ref');
    });
    if (compareCheckedIds.length > 0) {
        const refCard = grid.querySelector(`.compare-video-card[data-video-id="${compareCheckedIds[0]}"]`);
        if (refCard) refCard.classList.add('is-ref');
    }
}

const UNPLAYABLE_EXTENSIONS = ['.mxf', '.r3d', '.braw', '.ari'];

function isUnplayableFile(video) {
    const s3Key = video.s3_key || video.video_url || '';
    const ext = s3Key.substring(s3Key.lastIndexOf('.')).toLowerCase();
    return UNPLAYABLE_EXTENSIONS.includes(ext);
}
```

- [ ] **Step 2: Rewrite `loadCompareVideos()` with checkbox multi-select**

Replace the entire `loadCompareVideos()` function (lines 8167-8221) with:

```javascript
function loadCompareVideos() {
    const grid = document.getElementById('compareVideoGrid');
    if (!grid) return;

    // Reset selection
    compareCheckedIds = [];
    compareRefVideoId = null;
    updateCompareBtn();

    grid.innerHTML = '<div class="compare-video-grid-empty">Loading videos...</div>';

    fetch(`/api/indexes/${compareBackend}/${compareIndexMode}/videos`)
        .then(r => r.json())
        .then(data => {
            const videos = data.videos || data || [];
            grid.innerHTML = '';

            if (videos.length === 0) {
                grid.innerHTML = '<div class="compare-video-grid-empty">No indexed videos found. Upload and process videos first.</div>';
                return;
            }

            videos.forEach(v => {
                if (v.video_url) compareVideoUrls[v.video_id] = v.video_url;
                const card = document.createElement('div');
                card.className = 'compare-video-card';
                card.dataset.videoId = v.video_id;

                const unplayable = isUnplayableFile(v);
                const ext = (v.s3_key || v.video_url || '').split('.').pop().toUpperCase();
                const techVideo = v.technical_metadata?.video;
                const codecTag = techVideo ? [techVideo.codec, techVideo.resolution || (techVideo.width && techVideo.height ? techVideo.width + 'x' + techVideo.height : '')].filter(Boolean).join(' · ') : '';

                if (v.video_url && !unplayable) {
                    card.innerHTML = `
                        <span class="card-checkbox">&#10003;</span>
                        <span class="card-ref-badge">REF</span>
                        <video src="${escapeHtml(v.video_url)}" muted preload="metadata"></video>
                        <div class="card-name">${escapeHtml(v.name || v.video_id)}</div>
                        ${codecTag ? `<div class="card-codec">${escapeHtml(codecTag)}</div>` : ''}
                    `;
                    const vid = card.querySelector('video');
                    card.addEventListener('mouseenter', () => { vid.currentTime = 0; vid.play().catch(() => {}) });
                    card.addEventListener('mouseleave', () => { try { vid.pause(); vid.currentTime = 0; } catch(e){} });
                } else {
                    card.innerHTML = `
                        <span class="card-checkbox">&#10003;</span>
                        <span class="card-ref-badge">REF</span>
                        <div style="aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;background:var(--bg-deep);color:var(--text-tertiary);font-size:20px">&#9654;</div>
                        <div class="card-name">${escapeHtml(v.name || v.video_id)}</div>
                        ${unplayable ? `<div class="card-warning">${ext} · no proxy</div>` : ''}
                        ${codecTag ? `<div class="card-codec">${escapeHtml(codecTag)}</div>` : ''}
                    `;
                }

                // Toggle checkbox on click
                card.addEventListener('click', () => {
                    const idx = compareCheckedIds.indexOf(v.video_id);
                    if (idx >= 0) {
                        // Uncheck
                        compareCheckedIds.splice(idx, 1);
                        card.classList.remove('selected');
                    } else {
                        // Check — cap at 9 (1 ref + 8 compare)
                        if (compareCheckedIds.length >= 9) {
                            alert('Maximum 8 comparison videos (9 total including reference).');
                            return;
                        }
                        compareCheckedIds.push(v.video_id);
                        card.classList.add('selected');
                    }
                    compareRefVideoId = compareCheckedIds[0] || null;
                    updateRefBadge();
                    updateCompareBtn();
                });

                grid.appendChild(card);
            });
        })
        .catch(err => {
            console.error('Failed to load compare videos:', err);
            grid.innerHTML = '<div class="compare-video-grid-empty">Failed to load videos.</div>';
        });
}
```

- [ ] **Step 3: Add `compareCheckedVideos()` function**

Add after `loadCompareVideos()`:

```javascript
function compareCheckedVideos() {
    if (compareCheckedIds.length < 2) return;
    const refId = compareCheckedIds[0];
    const cmpIds = compareCheckedIds.slice(1);

    // Hide the results list if visible, show diff container
    document.getElementById('compareResultsList').style.display = 'none';

    showTimelineDiff(refId, cmpIds, []);
}
```

- [ ] **Step 4: Verify and commit**

Open Compare page:
- Verify video cards show checkboxes (top-left)
- Click cards — they toggle checked/unchecked
- First checked card gets REF badge
- Compare button shows count, enables at ≥2
- MXF files show red warning tag
- Clicking Compare loads the NLE timeline

```bash
git add static/index.html
git commit -m "feat(compare): checkbox multi-select replacing 2-step selection flow"
```

---

### Task 3: Rewrite Video Preview Strip with Thumbnails and Mini Players

**Files:**
- Modify: `static/index.html:3676-3723` (`.tl-selector-strip` and `.tl-video-card` CSS)
- Modify: `static/index.html:8893-8946` (`renderTlSelectorStrip` function)

- [ ] **Step 1: Update video card CSS for 200px width with thumbnail**

Replace the `.tl-video-card` CSS (lines ~3685-3723) with:

```css
.tl-video-card {
    flex: 0 0 200px;
    border-radius: 8px;
    background: var(--bg-elevated);
    cursor: pointer;
    transition: all 0.15s;
    position: relative;
    overflow: hidden;
    border: 2px solid var(--border-light);
}
.tl-video-card:hover { border-color: var(--border-hover); }
.tl-video-card.selected { border-color: var(--accent); }
.tl-video-card .tl-card-thumb {
    position: relative;
    height: 112px;
    background: var(--bg-deep);
    overflow: hidden;
}
.tl-video-card .tl-card-thumb video,
.tl-video-card .tl-card-thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.tl-video-card .tl-card-play {
    position: absolute;
    bottom: 6px;
    left: 6px;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: rgba(255,255,255,0.2);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 8px;
    color: white;
    cursor: pointer;
    border: none;
    transition: background 0.15s;
    z-index: 2;
}
.tl-video-card .tl-card-play:hover { background: rgba(255,255,255,0.35); }
.tl-video-card .tl-card-play.disabled {
    opacity: 0.4;
    cursor: not-allowed;
}
.tl-video-card .tl-card-progress {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: rgba(255,255,255,0.15);
}
.tl-video-card .tl-card-progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 0 2px 2px 0;
    width: 0%;
    transition: width 0.2s;
}
.tl-video-card .tl-card-duration {
    position: absolute;
    bottom: 6px;
    right: 6px;
    font-size: 8px;
    color: rgba(255,255,255,0.7);
    font-family: 'IBM Plex Mono', monospace;
    background: rgba(0,0,0,0.5);
    padding: 1px 4px;
    border-radius: 2px;
}
.tl-video-card .tl-card-info {
    padding: 6px 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.tl-video-card .tl-card-name {
    font-size: 10px;
    color: var(--text-primary);
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.tl-video-card .tl-card-meta {
    font-size: 8px;
    color: var(--text-tertiary);
    font-family: 'IBM Plex Mono', monospace;
}
```

- [ ] **Step 2: Rewrite `renderTlSelectorStrip()` with thumbnails and mini players**

Replace the entire `renderTlSelectorStrip()` function (lines ~8893-8946) with:

```javascript
function renderTlSelectorStrip(data) {
    const strip = document.getElementById('tlSelectorStrip');
    strip.innerHTML = '';

    // Initialize includedVideoIds with all compare videos
    tlState.includedVideoIds = data.comparisons.map(c => c.video_id);

    const videoUrls = data.video_urls || {};

    // Helper to build card
    function buildPreviewCard(videoId, name, techMeta, badge, badgeColor, borderColor) {
        const card = document.createElement('div');
        card.className = 'tl-video-card selected';
        card.style.borderColor = borderColor;

        const url = videoUrls[videoId];
        const s3Key = data.reference?.metadata?.s3_key || '';
        const ext = s3Key.substring(s3Key.lastIndexOf('.')).toLowerCase();
        const unplayable = UNPLAYABLE_EXTENSIONS.includes(ext);
        const codecTag = techMeta?.video ? [techMeta.video.codec, techMeta.video.resolution || (techMeta.video.width && techMeta.video.height ? techMeta.video.width + 'x' + techMeta.video.height : '')].filter(Boolean).join(' · ') : '';

        const thumbHtml = url && !unplayable
            ? `<video src="${escapeHtml(url)}" muted preload="metadata" style="width:100%;height:100%;object-fit:cover"></video>`
            : `<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:var(--text-tertiary);font-size:24px">&#9654;</div>`;

        card.innerHTML = `
            <div class="tl-card-thumb">
                ${thumbHtml}
                <button class="tl-card-play${unplayable || !url ? ' disabled' : ''}" ${unplayable ? 'title="File format not playable in browser"' : ''}>&#9654;</button>
                <div class="tl-card-progress"><div class="tl-card-progress-fill"></div></div>
                <span class="tl-card-duration">${tlFormatTime(data.reference?.metadata?.duration || 0)}</span>
            </div>
            <div class="tl-card-info">
                <span class="tl-badge" style="background:${badgeColor};color:#1D1C1B;font-size:8px;padding:1px 5px;border-radius:2px;font-weight:700;">${badge}</span>
                <span class="tl-card-name">${escapeHtml(name)}</span>
                ${codecTag ? `<span class="tl-card-meta">${escapeHtml(codecTag)}</span>` : ''}
            </div>
        `;

        // Play button handler
        const playBtn = card.querySelector('.tl-card-play');
        const videoEl = card.querySelector('video');
        if (playBtn && videoEl && !playBtn.classList.contains('disabled')) {
            playBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (videoEl.paused) {
                    videoEl.play().catch(() => {
                        // Player error fallback
                        playBtn.classList.add('disabled');
                        playBtn.title = 'Format not supported';
                    });
                    playBtn.innerHTML = '&#9646;&#9646;';
                } else {
                    videoEl.pause();
                    playBtn.innerHTML = '&#9654;';
                }
            });
            // Update progress bar
            videoEl.addEventListener('timeupdate', () => {
                const pct = videoEl.duration ? (videoEl.currentTime / videoEl.duration) * 100 : 0;
                card.querySelector('.tl-card-progress-fill').style.width = pct + '%';
            });
            // Player error fallback
            videoEl.addEventListener('error', () => {
                playBtn.classList.add('disabled');
                playBtn.title = 'Format not supported';
            });
        }

        return card;
    }

    // Reference card
    const refName = data.reference.metadata.name || data.reference.video_id;
    const refCard = buildPreviewCard(
        data.reference.video_id, refName,
        data.reference.technical_metadata,
        'REF', 'var(--accent)', 'var(--accent)'
    );
    // Reference card is not toggleable
    strip.appendChild(refCard);

    // Compare video cards
    data.comparisons.forEach((cmp, idx) => {
        const dotColor = TL_DOT_COLORS[idx % TL_DOT_COLORS.length];
        const cmpName = cmp.metadata.name || cmp.video_id;

        // Override s3_key check for compare videos
        const cmpCard = document.createElement('div');
        cmpCard.className = 'tl-video-card selected';
        cmpCard.style.borderColor = dotColor;

        const url = videoUrls[cmp.video_id];
        const cmpS3 = cmp.metadata?.s3_key || '';
        const cmpExt = cmpS3.substring(cmpS3.lastIndexOf('.')).toLowerCase();
        const cmpUnplayable = UNPLAYABLE_EXTENSIONS.includes(cmpExt);
        const cmpTech = cmp.technical_metadata;
        const cmpCodec = cmpTech?.video ? [cmpTech.video.codec, cmpTech.video.resolution || (cmpTech.video.width && cmpTech.video.height ? cmpTech.video.width + 'x' + cmpTech.video.height : '')].filter(Boolean).join(' · ') : '';

        const cmpThumbHtml = url && !cmpUnplayable
            ? `<video src="${escapeHtml(url)}" muted preload="metadata" style="width:100%;height:100%;object-fit:cover"></video>`
            : `<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:var(--text-tertiary);font-size:24px">&#9654;</div>`;

        const lang = cmp.alignment?.language_variant?.detected ? '<span class="tl-badge lang" style="margin-left:4px">LANG</span>' : '';

        cmpCard.innerHTML = `
            <div class="tl-card-thumb">
                ${cmpThumbHtml}
                <button class="tl-card-play${cmpUnplayable || !url ? ' disabled' : ''}" ${cmpUnplayable ? 'title="File format not playable in browser"' : ''}>&#9654;</button>
                <div class="tl-card-progress"><div class="tl-card-progress-fill"></div></div>
                <span class="tl-card-duration">${tlFormatTime(cmp.metadata.duration || 0)}</span>
            </div>
            <div class="tl-card-info">
                <span class="tl-badge" style="background:${dotColor};color:#1D1C1B;font-size:8px;padding:1px 5px;border-radius:2px;font-weight:700;">${idx + 2}</span>${lang}
                <span class="tl-card-name">${escapeHtml(cmpName)}</span>
                ${cmpCodec ? `<span class="tl-card-meta">${escapeHtml(cmpCodec)}</span>` : ''}
            </div>
        `;

        // Play button handler for compare card
        const playBtn = cmpCard.querySelector('.tl-card-play');
        const videoEl = cmpCard.querySelector('video');
        if (playBtn && videoEl && !playBtn.classList.contains('disabled')) {
            playBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (videoEl.paused) {
                    videoEl.play().catch(() => {
                        playBtn.classList.add('disabled');
                        playBtn.title = 'Format not supported';
                    });
                    playBtn.innerHTML = '&#9646;&#9646;';
                } else {
                    videoEl.pause();
                    playBtn.innerHTML = '&#9654;';
                }
            });
            videoEl.addEventListener('timeupdate', () => {
                const pct = videoEl.duration ? (videoEl.currentTime / videoEl.duration) * 100 : 0;
                cmpCard.querySelector('.tl-card-progress-fill').style.width = pct + '%';
            });
            videoEl.addEventListener('error', () => {
                playBtn.classList.add('disabled');
                playBtn.title = 'Format not supported';
            });
        }

        // Toggle inclusion on click (not on play button)
        cmpCard.addEventListener('click', () => {
            const i = tlState.includedVideoIds.indexOf(cmp.video_id);
            if (i >= 0) {
                tlState.includedVideoIds.splice(i, 1);
                cmpCard.classList.remove('selected');
                cmpCard.style.borderColor = 'var(--border-light)';
            } else {
                tlState.includedVideoIds.push(cmp.video_id);
                cmpCard.classList.add('selected');
                cmpCard.style.borderColor = dotColor;
            }
            renderTimeline();
        });

        strip.appendChild(cmpCard);
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

- [ ] **Step 3: Verify and commit**

Trigger a comparison. Verify:
- Preview strip shows 200px cards with video thumbnails
- Play buttons work (play/pause individual videos)
- MXF files show disabled play button
- Progress bar updates during playback
- Duration label shows
- Codec/resolution tag shows
- Color-coded borders match track dots

```bash
git add static/index.html
git commit -m "feat(compare): video preview strip with thumbnails and mini players"
```

---

## Chunk 3: Zoom + Debug Controls

### Task 4: Timeline Zoom Controls

**Files:**
- Modify: `static/index.html:8786-8796` (`tlState` — add `zoom` field)
- Modify: `static/index.html:8948-9001` (`renderTimeline` — apply zoom)
- Modify: `static/index.html:9098-9115` (`renderTlRuler` — zoom-aware)
- Modify: `static/index.html:9213-9240` (`updatePlayheadPosition` — zoom-aware)
- Add JS near line 9210 (zoom handlers)

- [ ] **Step 1: Add `zoom` field to `tlState`**

At line 8796, inside the `tlState` object, add:

```javascript
zoom: 1.0,             // timeline zoom level (0.5 to 5.0)
```

- [ ] **Step 2: Apply zoom to timeline container**

In `renderTimeline()` (line ~8948), after `renderTlRuler(duration);`, add:

```javascript
// Apply zoom to track bars
const timeline = document.getElementById('tlTimeline');
timeline.style.overflowX = tlState.zoom > 1 ? 'auto' : 'hidden';
```

In `buildRefTrack()`, after `bar.className = 'tl-track-bar';`, add:

```javascript
bar.style.width = (100 * tlState.zoom) + '%';
```

In `buildCmpTrack()`, after `bar.className = 'tl-track-bar';`, add:

```javascript
bar.style.width = (100 * tlState.zoom) + '%';
```

- [ ] **Step 3: Make ruler zoom-aware**

In `renderTlRuler()` (line ~9098), the ruler already uses `ruler.offsetWidth`. After zoom, the effective width is `rulerWidth * tlState.zoom`. Update the ruler container:

After `ruler.innerHTML = '';` add:

```javascript
ruler.style.width = (100 * tlState.zoom) + '%';
```

- [ ] **Step 4: Make playhead zoom-aware**

In `updatePlayheadPosition()` (line ~9213), the playhead position calculation needs to account for zoom. Find the line that calculates `leftPx`. The playhead position should multiply by zoom:

```javascript
// The effective track width is the displayed width * zoom
const trackWidth = (timeline.offsetWidth - LABEL_W) * tlState.zoom;
const leftPx = LABEL_W + (tlState.playheadTime / duration) * trackWidth;
```

- [ ] **Step 5: Add zoom handlers**

Add after the DOMContentLoaded threshold listener (around line 9210):

```javascript
function tlZoom(delta) {
    const slider = document.getElementById('tlZoomSlider');
    let val = tlState.zoom + delta;
    val = Math.max(0.5, Math.min(5, val));
    tlState.zoom = val;
    slider.value = Math.round(val * 100);
    document.getElementById('tlZoomVal').textContent = val.toFixed(1) + 'x';
    renderTimeline();
}

// Zoom slider
document.addEventListener('DOMContentLoaded', () => {
    const zoomSlider = document.getElementById('tlZoomSlider');
    if (zoomSlider) {
        zoomSlider.addEventListener('input', () => {
            tlState.zoom = parseInt(zoomSlider.value) / 100;
            document.getElementById('tlZoomVal').textContent = tlState.zoom.toFixed(1) + 'x';
            renderTimeline();
        });
    }
});

// Alt+scroll to zoom on timeline
document.addEventListener('DOMContentLoaded', () => {
    const timeline = document.getElementById('tlTimeline');
    if (timeline) {
        timeline.addEventListener('wheel', (e) => {
            if (!e.altKey) return;
            e.preventDefault();
            const delta = e.deltaY > 0 ? -0.25 : 0.25;
            tlZoom(delta);
        }, { passive: false });
    }
});
```

- [ ] **Step 6: Make consensus heatmap zoom-aware**

In `renderTlConsensus()`, find where the consensus bar container is created and add the same zoom width:

```javascript
// After creating the bar container element:
barContainer.style.width = (100 * tlState.zoom) + '%';
```

- [ ] **Step 7: Seek all video players on playhead snap**

In `snapPlayheadToSegment()` (line ~9255), after `updatePlayheadPosition();`, add:

```javascript
// Seek all video players in preview strip to playhead time
document.querySelectorAll('#tlSelectorStrip video').forEach(vid => {
    try { vid.currentTime = tlState.playheadTime; } catch(e) {}
});
```

- [ ] **Step 8: Verify and commit**

Test zoom:
- Click +/− buttons — timeline expands/shrinks
- Drag zoom slider — smooth zoom
- Alt+scroll wheel on timeline — zooms in/out
- Ruler ticks stay aligned with track segments at all zoom levels
- Playhead position stays correct at all zoom levels
- Horizontal scrollbar appears at zoom >1x

```bash
git add static/index.html
git commit -m "feat(compare): timeline zoom controls with scroll-wheel support"
```

---

### Task 5: Debug Threshold and Modality Controls

**Files:**
- Modify: `static/index.html:8811-8818` (`getSegScore` — verify/fix)
- Modify: `static/index.html:8819-8831` (`segColorClass` — verify/fix)

- [ ] **Step 1: Investigate `getSegScore()`**

Read the current `getSegScore()` function (lines 8811-8818). The function looks correct logically. The issue may be that `seg.modality_scores` keys don't match `tlState.modalityMode` values. Check the API response shape:

- `tlState.modalityMode` values: `'combined'`, `'visual'`, `'audio'`, `'transcription'`
- `seg.modality_scores` keys from API: `'visual'`, `'audio'`, `'transcription'`

The function uses `ms[tlState.modalityMode]` — when mode is `'combined'` it returns `seg.similarity` (correct). When mode is `'visual'`/`'audio'`/`'transcription'` it returns `ms[mode]` (correct if keys match).

**Most likely bug**: The segment color class `segColorClass()` uses `getSegScore()` for the alert check, but the segment status class (`matched`/`changed`/`missing`) is based on `seg.status` (from the API), NOT on the modality-specific score. When switching modalities, the segment **background color** (which comes from the `matched`/`changed` CSS class) doesn't change — only the alert threshold changes.

**Fix**: When modalityMode is not 'combined', override the status class based on the modality-specific score:

```javascript
function segColorClass(seg) {
    const score = getSegScore(seg);
    let status = seg.status;
    // When a specific modality is selected, re-derive status from that score
    if (tlState.modalityMode !== 'combined' && score != null) {
        if (score >= 0.9) status = 'matched';
        else if (score >= 0.7) status = 'changed';
        // Don't override 'missing' or 'added' — those are structural, not score-based
        if (seg.status !== 'missing' && seg.status !== 'added') {
            // Only override matched/changed
        }
    }
    const classes = [status];
    if (seg.time_shift != null && Math.abs(seg.time_shift) > 0.5) classes.push('shifted');
    if (seg.status === 'missing') { classes.push('alert'); }
    else if (score != null && score < tlState.threshold / 100) { classes.push('alert'); }
    return classes.join(' ');
}
```

Wait — this needs more careful thought. The actual `segColorClass` at lines 8819-8831 is:

```javascript
function segColorClass(seg) {
    const classes = [seg.status];
    if (seg.time_shift != null && Math.abs(seg.time_shift) > 0.5) classes.push('shifted');
    if (seg.status === 'missing') { classes.push('alert'); }
    else {
        const score = getSegScore(seg);
        if (score != null && score < tlState.threshold / 100) classes.push('alert');
    }
    return classes.join(' ');
}
```

The issue is that `seg.status` (matched/changed) is always based on the **combined** similarity from the API. When the user switches to "Visual" modality, they expect to see segments re-colored based on visual-only scores. But `seg.status` doesn't change — it's fixed from the API response.

**The fix**: Derive status dynamically from the selected modality's score:

Replace `segColorClass()` with:

```javascript
function segColorClass(seg) {
    let status = seg.status;
    // Re-derive matched/changed status from selected modality score
    if (status !== 'missing' && status !== 'added' && tlState.modalityMode !== 'combined') {
        const score = getSegScore(seg);
        if (score != null) {
            status = score >= 0.9 ? 'matched' : 'changed';
        }
    }
    const classes = [status];
    if (seg.time_shift != null && Math.abs(seg.time_shift) > 0.5) classes.push('shifted');
    if (seg.status === 'missing') { classes.push('alert'); }
    else {
        const score = getSegScore(seg);
        if (score != null && score < tlState.threshold / 100) classes.push('alert');
    }
    return classes.join(' ');
}
```

- [ ] **Step 2: Also update tooltip text to use modality score**

In `buildCmpTrack()` (around line 9079), the tooltip uses `getSegScore(seg)` which already respects modality mode. Verify this is the case — it should be correct.

- [ ] **Step 3: Verify and commit**

Test:
- Switch to "Visual" modality — segment colors should change if visual scores differ from combined
- Switch to "Audio" — same behavior
- Drag threshold slider — segments below threshold get pulsing red outline
- "Diffs Only" filter hides green segments
- "Shifts" filter shows only time-shifted segments

```bash
git add static/index.html
git commit -m "fix(compare): modality pills now re-derive segment status from selected modality score"
```

---

## Verification

After all tasks, open the app and test the full flow end-to-end:

1. Navigate to Compare page
2. All buttons use toggle style (muted green, not solid)
3. Video grid shows checkboxes — check 3 videos
4. First checked gets REF badge
5. Click "Compare (3)"
6. NLE timeline loads — preview strip shows 200px cards with thumbnails
7. Play button works on playable videos
8. MXF files show disabled play button
9. Zoom slider works — timeline expands with horizontal scroll
10. Alt+scroll zooms
11. Switch modality pills — segment colors change
12. Drag threshold — alert outlines appear/disappear
13. Segment legend visible below stats bar
14. Filter tooltips show on hover
15. Playhead snaps to segments and seeks video players

Push to `feature/nav-redesign-compare-upload` branch, deploy to staging:
```bash
git push origin feature/nav-redesign-compare-upload
aws apprunner start-deployment --service-arn "arn:aws:apprunner:us-east-1:026090552520:service/video-search-staging/f5e755cb7c304fe4ba1b18a0b6154992" --region us-east-1
```
