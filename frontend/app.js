const API_BASE_URL = 'http://127.0.0.1:8000';

const el = {
    dropzone:    document.getElementById('dropzone'),
    fileInput:   document.getElementById('fileInput'),
    fileName:    document.getElementById('fileName'),
    runBtn:      document.getElementById('runBtn'),
    runBtnText:  document.getElementById('runBtnText'),
    runBtnSpinner: document.getElementById('runBtnSpinner'),
    runBtnArrow: document.getElementById('runBtnArrow'),
    loader:      document.getElementById('loader'),
    results:     document.getElementById('resultsSection'),
    error:       document.getElementById('errorMessage'),
    statusDot:   document.getElementById('statusDot'),
    stats: {
        total:  document.getElementById('statTotalSlices'),
        tumor:  document.getElementById('statTumorSlices'),
        volume: document.getElementById('statTumorVolume')
    },
    overlayStrip: document.getElementById('overlayStrip')
};

let selectedFile = null;

// ─────────────────────────────────────────
// INIT
// ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupDropzone();
    el.runBtn.addEventListener('click', runSegmentation);
});

// ─────────────────────────────────────────
// HEALTH CHECK
// ─────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
            el.statusDot.classList.remove('offline');
            el.statusDot.classList.add('online');
        } else {
            el.statusDot.classList.remove('online');
            el.statusDot.classList.add('offline');
        }
    } catch (e) {
        el.statusDot.classList.remove('online');
        el.statusDot.classList.add('offline');
    }
}

// ─────────────────────────────────────────
// DROPZONE
// ─────────────────────────────────────────
function setupDropzone() {
    el.dropzone.addEventListener('click', () => el.fileInput.click());

    el.fileInput.addEventListener('change', (e) => {
        handleFile(e.target.files[0]);
    });

    el.dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        el.dropzone.classList.add('dragover');
    });

    el.dropzone.addEventListener('dragleave', () => {
        el.dropzone.classList.remove('dragover');
    });

    el.dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        el.dropzone.classList.remove('dragover');
        handleFile(e.dataTransfer.files[0]);
    });
}

function handleFile(file) {
    if (!file || !file.name.endsWith('.zip')) {
        showError('Please select a ZIP file');
        return;
    }
    selectedFile = file;
    el.fileName.textContent = file.name;
    el.runBtn.disabled = false;
    hideError();
}

// ─────────────────────────────────────────
// RUN SEGMENTATION
// ─────────────────────────────────────────
async function runSegmentation() {
    if (!selectedFile) return;

    // Button loading state
    setButtonLoading(true);
    el.results.classList.add('hidden');
    hideError();
    el.loader.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const res = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || 'Segmentation failed');
        }

        renderResults(data);
        showToast('Analysis complete ✓', 'success');

    } catch (e) {
        showError(e.message);
        showToast(e.message, 'error');
    } finally {
        el.loader.classList.add('hidden');
        setButtonLoading(false);
    }
}

function setButtonLoading(loading) {
    el.runBtn.disabled = loading;
    if (el.runBtnText)    el.runBtnText.textContent    = loading ? 'Analysing...' : 'Run Segmentation';
    if (el.runBtnSpinner) el.runBtnSpinner.style.display = loading ? 'block' : 'none';
    if (el.runBtnArrow)   el.runBtnArrow.style.display   = loading ? 'none'  : 'inline';
    el.runBtn.style.background = loading ? 'var(--surface)' : '';
    el.runBtn.style.color      = loading ? 'var(--text2)'   : '';
    el.runBtn.style.cursor     = loading ? 'not-allowed'    : '';
}

// ─────────────────────────────────────────
// RENDER RESULTS
// ─────────────────────────────────────────
function renderResults(data) {
    // Safe fallbacks for missing/null values
    const totalSlices  = data.total_slices  ?? 0;
    const tumorSlices  = data.tumor_slices  ?? 0;
    const tumorVolume  = (data.total_tumor_volume != null)
        ? parseFloat(data.total_tumor_volume).toFixed(2)
        : '—';
    const overlays     = data.overlays ?? [];
    const tumorSliceIds = data.tumor_slice_ids ?? [];

    el.stats.total.textContent  = totalSlices;
    el.stats.tumor.textContent  = tumorSlices;
    el.stats.volume.textContent = tumorVolume;

    // Build slice density bar
    buildDensityBar(tumorSliceIds, totalSlices);

    // Build overlay cards
    el.overlayStrip.innerHTML = '';
    overlays.forEach(item => {
        const card = document.createElement('div');
        card.className = 'overlay-card';
        card.dataset.pixels = item.tumor_pixels ?? 0;

        card.innerHTML = `
            <img src="data:image/png;base64,${item.image_base64}" alt="Slice ${item.slice_index}" loading="lazy">
            <div class="overlay-zoom-hint">🔍</div>
            <div class="overlay-info">
                <span class="slice-idx">Slice ${item.slice_index}</span>
                <span class="pixel-count">${item.tumor_pixels ?? 0} tumor pixels</span>
            </div>
        `;

        card.addEventListener('click', () => {
            openLightbox(
                `data:image/png;base64,${item.image_base64}`,
                item.slice_index,
                item.tumor_pixels ?? 0,
                card.classList.contains('top-slice')
            );
        });

        el.overlayStrip.appendChild(card);
    });

    // Reveal results with smooth scroll after short delay
    el.results.classList.remove('hidden');
    setTimeout(() => {
        el.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 120);
}

// ─────────────────────────────────────────
// SLICE DENSITY BAR
// ─────────────────────────────────────────
function buildDensityBar(tumorSliceIds, totalSlices) {
    const bar = document.getElementById('densityBar');
    if (!bar || totalSlices === 0) return;

    const tumorSet = new Set(tumorSliceIds);
    bar.innerHTML = '';

    for (let i = 0; i < totalSlices; i++) {
        const seg = document.createElement('div');
        seg.className = 'density-seg' + (tumorSet.has(i) ? ' tumor' : '');
        seg.title = `Slice ${i}${tumorSet.has(i) ? ' — tumor detected' : ''}`;
        bar.appendChild(seg);
    }
}

// ─────────────────────────────────────────
// LIGHTBOX
// ─────────────────────────────────────────
function openLightbox(src, sliceIdx, pixels, isTop) {
    document.getElementById('lightboxImg').src     = src;
    document.getElementById('lightboxSlice').textContent  = `Slice ${sliceIdx}`;
    document.getElementById('lightboxPixels').textContent = `${pixels} tumor pixels`;
    const badge = document.getElementById('lightboxBadge');
    badge.textContent  = isTop ? '★ Highest density' : 'Detection';
    badge.style.color  = isTop ? '#ffb400' : 'var(--accent)';
    document.getElementById('lightbox').classList.remove('hidden');
}

function closeLightbox() {
    document.getElementById('lightbox').classList.add('hidden');
}

// Escape key closes lightbox
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeLightbox();
});

// ─────────────────────────────────────────
// TOAST
// ─────────────────────────────────────────
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    // Trigger animation
    requestAnimationFrame(() => toast.classList.add('show'));

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 400);
    }, 3200);
}

// ─────────────────────────────────────────
// ERROR
// ─────────────────────────────────────────
function showError(msg) {
    el.error.textContent = msg || 'An unexpected error occurred.';
    el.error.classList.remove('hidden');
}

function hideError() {
    el.error.classList.add('hidden');
}
