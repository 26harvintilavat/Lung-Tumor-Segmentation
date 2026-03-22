const API_BASE_URL = 'http://localhost:8000';

const el = {
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('fileInput'),
    fileName: document.getElementById('fileName'),
    runBtn: document.getElementById('runBtn'),
    loader: document.getElementById('loader'),
    results: document.getElementById('resultsSection'),
    error: document.getElementById('errorMessage'),
    statusDot: document.getElementById('statusDot'),
    stats: {
        total: document.getElementById('statTotalSlices'),
        tumor: document.getElementById('statTumorSlices'),
        volume: document.getElementById('statTumorVolume')
    },
    overlayStrip: document.getElementById('overlayStrip')
};

let selectedFile = null;

// INIT
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupDropzone();
    el.runBtn.addEventListener('click', runSegmentation);
});

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
            el.statusDot.classList.add('online');
        } else {
            el.statusDot.classList.add('offline');
        }
    } catch (e) {
        el.statusDot.classList.add('offline');
    }
}

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

async function runSegmentation() {
    if (!selectedFile) return;

    // Reset UI
    el.runBtn.disabled = true;
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
    } catch (e) {
        showError(e.message);
    } finally {
        el.loader.classList.add('hidden');
        el.runBtn.disabled = false;
    }
}

function renderResults(data) {
    el.stats.total.textContent = data.total_slices;
    el.stats.tumor.textContent = data.tumor_slices;
    el.stats.volume.textContent = data.total_tumor_volume.toFixed(2);

    el.overlayStrip.innerHTML = '';
    data.overlays.forEach(item => {
        const card = document.createElement('div');
        card.className = 'overlay-card';
        card.innerHTML = `
            <img src="data:image/png;base64,${item.image_base64}" alt="Slice ${item.slice_index}">
            <div class="overlay-info">
                <span class="slice-idx">Slice ${item.slice_index}</span>
                <span class="pixel-count">${item.tumor_pixels} tumor pixels</span>
            </div>
        `;
        el.overlayStrip.appendChild(card);
    });

    el.results.classList.remove('hidden');
    el.results.scrollIntoView({ behavior: 'smooth' });
}

function showError(msg) {
    el.error.textContent = msg;
    el.error.classList.remove('hidden');
}

function hideError() {
    el.error.classList.add('hidden');
}
