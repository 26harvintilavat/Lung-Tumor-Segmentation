// ==========================================
// LungSeg AI - Vanilla JS Frontend Logic
// ==========================================

// --- CONFIGURATION ---
const API_BASE_URL = 'http://localhost:8000';

// --- STATE ---
let currentPatientId = null;
let currentJobId = null;
let uploadedFiles = [];
let sliceDataList = []; // Array of object holding DICOM imageIds for Cornerstone
let currentSliceIndex = 0;
let maskDataArray = null; // Storing the full numpy flat array equivalent
let maskOpacity = 0.45;
let isMaskVisible = true;
let isPipelineRunning = false;
let pollingInterval = null;

// --- DOM ELEMENTS (Cached) ---
const el = {
    // Nav
    navLinks: document.querySelectorAll('.nav-link'),
    pages: document.querySelectorAll('.page-section'),
    backendBanner: document.getElementById('backendBanner'),
    toastContainer: document.getElementById('toastContainer'),

    // Upload
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('fileInput'),
    fileList: document.getElementById('fileList'),
    dropzoneContent: document.getElementById('dropzoneContent'),
    patientIdInput: document.getElementById('patientId'),
    presetSelect: document.getElementById('presetSelect'),
    customWindowing: document.getElementById('customWindowing'),
    customWW: document.getElementById('customWW'),
    customWL: document.getElementById('customWL'),
    runSegmentationBtn: document.getElementById('runSegmentationBtn'),
    uploadLoader: document.getElementById('uploadLoader'),
    loaderText: document.getElementById('loaderText'),

    // Viewer
    thumbnailContainer: document.getElementById('thumbnailContainer'),
    dicomViewport: document.getElementById('dicomViewport'),
    maskCanvas: document.getElementById('maskCanvas'),
    sliceScrubber: document.getElementById('sliceScrubber'),
    sliceIndicator: document.getElementById('sliceIndicator'),
    prevSliceBtn: document.getElementById('prevSliceBtn'),
    nextSliceBtn: document.getElementById('nextSliceBtn'),
    toggleMaskBtn: document.getElementById('toggleMaskBtn'),
    maskOpacitySlider: document.getElementById('maskOpacity'),
    resetViewBtn: document.getElementById('resetViewBtn'),
    invertBtn: document.getElementById('invertBtn'),
    wwDisplay: document.getElementById('wwDisplay'),
    wlDisplay: document.getElementById('wlDisplay'),

    // Stats
    v_patientId: document.getElementById('v_patientId'),
    v_seriesUid: document.getElementById('v_seriesUid'),
    v_totalSlices: document.getElementById('v_totalSlices'),
    v_thickness: document.getElementById('v_thickness'),
    v_nodules: document.getElementById('v_nodules'),
    v_volume: document.getElementById('v_volume'),
    v_diameter: document.getElementById('v_diameter'),
    v_range: document.getElementById('v_range'),
    v_confidenceVal: document.getElementById('v_confidenceVal'),
    v_confidenceBar: document.getElementById('v_confidenceBar'),
    downloadMaskBtn: document.getElementById('downloadMaskBtn'),
    downloadReportBtn: document.getElementById('downloadReportBtn'),
    historyTbody: document.getElementById('historyTbody'),

    // Pipeline
    pipelineSteps: document.querySelectorAll('#pipelineSteps .step'),
    terminalConsole: document.getElementById('terminalConsole'),
    rerunPipelineBtn: document.getElementById('rerunPipelineBtn')
};


// ==========================================
// INITIALIZATION
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    initRouting();
    initCornerstone();
    initUploadEvents();
    initViewerEvents();
    initPipelineEvents();
    loadRunHistory();
    checkBackendStatus();
});

// --- Hash Routing ---
function initRouting() {
    window.addEventListener('hashchange', handleRouteChange);

    // Initial load
    if (!window.location.hash) {
        window.location.hash = '#home';
    } else {
        handleRouteChange();
    }
}

function handleRouteChange() {
    const hash = window.location.hash.slice(1) || 'home';

    // Update nav links
    el.navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${hash}`) link.classList.add('active');
    });

    // Toggle pages
    el.pages.forEach(page => page.classList.add('hidden'));
    const targetPage = document.getElementById(`page-${hash}`);
    if (targetPage) {
        targetPage.classList.remove('hidden');

        // Viewer resize fix
        if (hash === 'viewer') {
            setTimeout(() => {
                const element = el.dicomViewport;
                if (cornerstone.getEnabledElements().find(e => e.element === element)) {
                    cornerstone.resize(element);
                    drawMaskOverlay(); // Redraw mask on resize
                }
            }, 100);
        }

    } else {
        document.getElementById('page-home').classList.remove('hidden'); // fallback
    }
}

// --- Cornerstone Init ---
function initCornerstone() {
    cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
    cornerstoneWADOImageLoader.external.dicomParser = dicomParser;

    // Initialize WebWorkers for faster decoding (required by wado loader)
    cornerstoneWADOImageLoader.webWorkerManager.initialize({
        maxWebWorkers: navigator.hardwareConcurrency || 1,
        startWebWorkersOnDemand: true,
        taskConfiguration: {
            decodeTask: { initializeCodecsOnStartup: false }
        }
    });

    cornerstone.enable(el.dicomViewport);
}

// --- Health Check ---
async function checkBackendStatus() {
    try {
        // Assume root endpoint exists for simple health check
        await fetch(`${API_BASE_URL}/docs`, { mode: 'no-cors' });
        el.backendBanner.classList.add('hidden');
    } catch (e) {
        el.backendBanner.classList.remove('hidden');
        console.warn("Backend might be offline.");
    }
}


// ==========================================
// UI UTILITIES
// ==========================================

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    el.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function addLogLine(text, type = 'normal') {
    const now = new Date();
    const timeString = `[${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}]`;
    const div = document.createElement('div');
    div.className = `log-line ${type}`;
    div.textContent = `${timeString} ${text}`;
    el.terminalConsole.appendChild(div);
    el.terminalConsole.scrollTop = el.terminalConsole.scrollHeight;
}


// ==========================================
// UPLOAD LOGIC
// ==========================================

function initUploadEvents() {
    el.dropzone.addEventListener('click', () => el.fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
        el.dropzone.addEventListener(evt, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(evt => {
        el.dropzone.addEventListener(evt, () => el.dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(evt => {
        el.dropzone.addEventListener(evt, () => el.dropzone.classList.remove('dragover'), false);
    });

    el.dropzone.addEventListener('drop', handleDrop, false);
    el.fileInput.addEventListener('change', handleFileSelect, false);

    el.presetSelect.addEventListener('change', (e) => {
        if (e.target.value === 'custom') {
            el.customWindowing.classList.remove('hidden');
        } else {
            el.customWindowing.classList.add('hidden');
        }
    });

    el.runSegmentationBtn.addEventListener('click', startPipeline);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    processFiles(files);
}

function handleFileSelect(e) {
    processFiles(e.target.files);
}

async function processFiles(files) {
    if (!files || files.length === 0) return;

    uploadedFiles = Array.from(files);

    // Sort files alphabetically to handle multiple slices (assuming naming convention defines order)
    uploadedFiles.sort((a, b) => a.name.localeCompare(b.name));

    el.dropzone.classList.add('has-files');
    el.dropzoneContent.classList.add('hidden');
    el.fileList.classList.remove('hidden');

    el.fileList.innerHTML = uploadedFiles.map(f => `
        <div class="file-item">
            <span>${f.name}</span>
            <span>${(f.size / 1024 / 1024).toFixed(2)} MB</span>
        </div>
    `).join('');

    el.runSegmentationBtn.disabled = false;

    // Try parsing the first file header locally to populate hints
    if (uploadedFiles[0].name.endsWith('.dcm')) {
        try {
            const arrayBuffer = await uploadedFiles[0].arrayBuffer();
            const byteArray = new Uint8Array(arrayBuffer);
            const dataSet = dicomParser.parseDicom(byteArray);

            const extractedPatId = dataSet.string('x00100020');
            if (extractedPatId && !el.patientIdInput.value) {
                el.patientIdInput.value = extractedPatId;
            }
        } catch (err) {
            console.warn("Could not parse DICOM locally:", err);
        }
    }
}


// ==========================================
// PIPELINE EXECUTION (API INTEGRATION)
// ==========================================

async function startPipeline() {
    if (uploadedFiles.length === 0) return;

    // Show UI overlay
    el.uploadLoader.classList.remove('hidden');
    el.loaderText.textContent = "Uploading CT scan...";

    let providedPatientId = el.patientIdInput.value.trim();
    if (!providedPatientId) {
        providedPatientId = `PT-${new Date().getTime().toString().slice(-6)}`;
    }
    currentPatientId = providedPatientId;

    try {
        // Step 1: POST /upload
        let formData = new FormData();
        // Send first file as representative, or loop to send all if backend supports multi-file upload.
        // For simplicity requested, sending one file or a zipped folder is common. We'll send the first.
        formData.append("file", uploadedFiles[0]);
        formData.append("patient_id", currentPatientId);

        const uploadRes = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!uploadRes.ok) {
            const errJson = await uploadRes.json();
            throw new Error(errJson.detail || errJson.message || "Upload Failed");
        }

        const metadata = await uploadRes.json();
        addLogLine(`Upload complete. Local patient_id: ${currentPatientId}`);

        // Update UI Text
        el.loaderText.textContent = "Triggering Segmentation Pipeline...";

        // Step 2: POST /segment
        const preset = getSelectedWindowing();
        const segmentRes = await fetch(`${API_BASE_URL}/segment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                patient_id: currentPatientId,
                window_width: preset.ww,
                window_level: preset.wl
            })
        });

        if (!segmentRes.ok) {
            const errJson = await segmentRes.json();
            throw new Error(errJson.detail || "Segmentation Trigger Failed");
        }

        const segmentData = await segmentRes.json();
        currentJobId = segmentData.job_id;

        // Route to Pipeline page to watch status
        el.uploadLoader.classList.add('hidden');
        window.location.hash = '#pipeline';
        startPollingStatus(currentJobId);

    } catch (error) {
        el.uploadLoader.classList.add('hidden');
        showToast(error.message, 'error');
        console.error(error);
    }
}

function getSelectedWindowing() {
    const val = el.presetSelect.value;
    if (val === 'lung') return { ww: 1500, wl: -600 };
    if (val === 'mediastinal') return { ww: 400, wl: 40 };
    if (val === 'bone') return { ww: 2000, wl: 400 };
    return {
        ww: parseInt(el.customWW.value) || 1500,
        wl: parseInt(el.customWL.value) || -600
    };
}

function startPollingStatus(jobId) {
    if (pollingInterval) clearInterval(pollingInterval);
    isPipelineRunning = true;

    // Reset Stepper
    el.pipelineSteps.forEach(s => {
        s.className = 'step pending';
        s.innerHTML = `<span class="icon">⏳</span> ${s.textContent.replace(/[⏳✅🔄❌]/, '').trim()}`;
    });

    addLogLine(`Polling status for Job: ${jobId}`, 'system');

    pollingInterval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/status/${jobId}`);
            if (!res.ok) throw new Error("Failed to fetch status");

            const data = await res.json();
            updatePipelineUI(data);

            if (data.status === 'complete') {
                clearInterval(pollingInterval);
                isPipelineRunning = false;
                showToast("Segmentation Complete!", "success");
                addLogLine("Pipeline finished successfully. redirecting...");

                setTimeout(() => {
                    loadResults(currentPatientId);
                }, 1000);
            }
            else if (data.status === 'failed') {
                clearInterval(pollingInterval);
                isPipelineRunning = false;
                showToast(`Pipeline Failed: ${data.error}`, "error");
                addLogLine(`Fatal Error: ${data.error}`, 'error');
            }

        } catch (e) {
            console.error(e);
            clearInterval(pollingInterval);
            showToast("Connection to backend lost during polling.", "error");
        }
    }, 2000);
}

function updatePipelineUI(statusData) {
    // Expected structure: { steps: [{name:'DICOM Upload', status: 'complete'}, ...], logs: [...] }
    if (!statusData.steps) return;

    statusData.steps.forEach((stepObj, index) => {
        const stepEl = el.pipelineSteps[index];
        if (!stepEl) return;

        const textNode = stepEl.textContent.replace(/[⏳✅🔄❌]/, '').trim();

        if (stepObj.status === 'complete') {
            stepEl.className = 'step complete';
            stepEl.innerHTML = `<span class="icon">✅</span> ${textNode}`;
        } else if (stepObj.status === 'running') {
            stepEl.className = 'step running';
            stepEl.innerHTML = `<span class="icon">🔄</span> ${textNode}`;
        } else if (stepObj.status === 'failed') {
            stepEl.className = 'step failed';
            stepEl.innerHTML = `<span class="icon">❌</span> ${textNode}`;
        }
    });

    // Update Logs (Assume backend returns only new logs or we clear and rewrite)
    if (statusData.logs && statusData.logs.length > 0) {
        // Simple approach: clear and redraw or implement differential logging.
        // Assuming backend gives all logs for the job.
        el.terminalConsole.innerHTML = '<div class="log-line system">LungSeg AI backend connection established.</div>';
        statusData.logs.forEach(msg => addLogLine(msg));
    }
}

function initPipelineEvents() {
    el.rerunPipelineBtn.addEventListener('click', () => {
        if (!currentPatientId) {
            showToast("No active patient to re-run.", "warning");
            return;
        }
        if (!isPipelineRunning) {
            window.location.hash = '#upload';
            // Pre-fill
            el.patientIdInput.value = currentPatientId;
        } else {
            showToast("Pipeline is already running.", "warning");
        }
    });
}


// ==========================================
// VIEWER & RESULTS RENDERING
// ==========================================

async function loadResults(patientId) {
    window.location.hash = '#viewer';
    addLogLine(`Fetching results for ${patientId}...`, 'system');

    try {
        const res = await fetch(`${API_BASE_URL}/results/${patientId}`);
        if (!res.ok) throw new Error("Could not fetch results data.");

        const data = await res.json();

        // Populate Stats Panel
        el.v_patientId.textContent = patientId;
        el.v_seriesUid.textContent = data.series_uid || 'N/A';
        el.v_totalSlices.textContent = data.total_slices || '--';
        el.v_thickness.textContent = data.slice_thickness ? `${data.slice_thickness} mm` : '--';

        el.v_nodules.textContent = data.nodules_detected || 0;
        el.v_volume.textContent = data.tumor_volume_cm3 ? `${data.tumor_volume_cm3} cm³` : '--';
        el.v_diameter.textContent = data.max_diameter_mm ? `${data.max_diameter_mm} mm` : '--';
        el.v_range.textContent = data.slice_range || '--';

        if (data.confidence_score) {
            el.v_confidenceVal.textContent = `${data.confidence_score}%`;
            el.v_confidenceBar.style.width = `${data.confidence_score}%`;
            if (data.confidence_score > 80) el.v_confidenceBar.style.backgroundColor = 'var(--success)';
            else if (data.confidence_score > 50) el.v_confidenceBar.style.backgroundColor = 'var(--warning)';
            else el.v_confidenceBar.style.backgroundColor = 'var(--error)';
        }

        // Save to History
        saveRunHistory({
            patientId: patientId,
            date: new Date().toLocaleDateString(),
            nodules: data.nodules_detected || 0,
            volume: data.tumor_volume_cm3 || 0,
            status: 'Success'
        });

        // Initialize Cornerstone Viewport Local state
        setupViewer(data);

    } catch (e) {
        showToast(e.message, 'error');
        console.error(e);
    }
}

async function setupViewer(backendData) {
    if (uploadedFiles.length === 0) {
        showToast("No local CT files available in memory to display.", "warning");
        return;
    }

    // 1. Prepare imageIds for Cornerstone WADO Loader
    sliceDataList = [];
    uploadedFiles.forEach(file => {
        const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file);
        sliceDataList.push(imageId);
    });

    // Assume one file contains multple frames if it's one file. If multiple dcm files, it's one slice per file.
    let totalFrames = sliceDataList.length;

    // Load first image to initialize
    try {
        const image = await cornerstone.loadImage(sliceDataList[0]);
        cornerstone.displayImage(el.dicomViewport, image);

        // Set initial windowing
        const preset = getSelectedWindowing();
        const viewport = cornerstone.getViewport(el.dicomViewport);
        viewport.voi.windowWidth = preset.ww;
        viewport.voi.windowCenter = preset.wl;
        cornerstone.setViewport(el.dicomViewport, viewport);

        updateWindowingDisplay(viewport);

        // Setup scrubber
        el.sliceScrubber.min = 1;
        el.sliceScrubber.max = totalFrames;
        el.sliceScrubber.value = 1;

        // Mask Data (Assuming Backend provides base64 flat numpy array mimicking the volume)
        // Format: Array of arrays (1 per slice) or a single 1D array we chunk.
        // For this frontend wrapper, we expect `backendData.mask_base64_json` or similar list representation.
        // Mocking empty mask if not provided to prevent crashes
        if (backendData.mask_array_json) {
            maskDataArray = backendData.mask_array_json; // e.g., [[0,1,0...], [0,0,1...]]
        } else {
            console.warn("No mask data returned from backend. Using empty array.");
            maskDataArray = new Array(totalFrames).fill([]);
        }

        renderSlice(0);
        generateThumbnails();

    } catch (e) {
        console.error("Cornerstone Loading Error:", e);
        showToast("Failed to render DICOM image.", "error");
    }
}

function initViewerEvents() {
    // Scrubber
    el.sliceScrubber.addEventListener('input', (e) => {
        const index = parseInt(e.target.value) - 1;
        renderSlice(index);
    });

    // Arrows
    el.prevSliceBtn.addEventListener('click', () => {
        if (currentSliceIndex > 0) renderSlice(currentSliceIndex - 1);
    });

    el.nextSliceBtn.addEventListener('click', () => {
        if (currentSliceIndex < sliceDataList.length - 1) renderSlice(currentSliceIndex + 1);
    });

    // Keyboard
    document.addEventListener('keydown', (e) => {
        if (window.location.hash !== '#viewer') return;
        if (e.key === 'ArrowLeft') {
            if (currentSliceIndex > 0) renderSlice(currentSliceIndex - 1);
        }
        if (e.key === 'ArrowRight') {
            if (currentSliceIndex < sliceDataList.length - 1) renderSlice(currentSliceIndex + 1);
        }
    });

    // Mask Controls
    el.toggleMaskBtn.addEventListener('click', () => {
        isMaskVisible = !isMaskVisible;
        el.toggleMaskBtn.classList.toggle('active', isMaskVisible);
        el.toggleMaskBtn.textContent = isMaskVisible ? "Mask ON" : "Mask OFF";
        drawMaskOverlay();
    });

    el.maskOpacitySlider.addEventListener('input', (e) => {
        maskOpacity = parseFloat(e.target.value);
        drawMaskOverlay();
    });

    // Tools
    el.resetViewBtn.addEventListener('click', () => {
        const viewport = cornerstone.getViewport(el.dicomViewport);
        viewport.scale = 1.0;
        viewport.translation.x = 0;
        viewport.translation.y = 0;
        const preset = getSelectedWindowing();
        viewport.voi.windowWidth = preset.ww;
        viewport.voi.windowCenter = preset.wl;
        cornerstone.setViewport(el.dicomViewport, viewport);
        updateWindowingDisplay(viewport);
    });

    el.invertBtn.addEventListener('click', () => {
        const viewport = cornerstone.getViewport(el.dicomViewport);
        viewport.invert = !viewport.invert;
        cornerstone.setViewport(el.dicomViewport, viewport);
    });

    // Cornerstone internal events
    el.dicomViewport.addEventListener('cornerstoneimagerendered', (e) => {
        const viewport = e.detail.viewport;
        updateWindowingDisplay(viewport);

        // Sync canvas size
        const canvas = el.maskCanvas;
        const cornerstoneCanvas = document.querySelector('.cornerstone-canvas');
        if (cornerstoneCanvas) {
            canvas.width = cornerstoneCanvas.width;
            canvas.height = cornerstoneCanvas.height;
            drawMaskOverlay();
        }
    });

    // Cornerstone input mapping (Zoom/Pan)
    el.dicomViewport.addEventListener('wheel', (e) => {
        e.preventDefault();
        const viewport = cornerstone.getViewport(el.dicomViewport);
        if (e.deltaY < 0) { viewport.scale += 0.1; }
        else { viewport.scale -= 0.1; }
        if (viewport.scale < 0.1) viewport.scale = 0.1;
        cornerstone.setViewport(el.dicomViewport, viewport);
    });

    let isPanning = false;
    let lastMousePos = { x: 0, y: 0 };

    el.dicomViewport.addEventListener('mousedown', (e) => {
        if (e.button === 0) { // left click
            isPanning = true;
            lastMousePos = { x: e.clientX, y: e.clientY };
        }
    });

    document.addEventListener('mouseup', () => isPanning = false);

    document.addEventListener('mousemove', (e) => {
        if (isPanning && window.location.hash === '#viewer') {
            const viewport = cornerstone.getViewport(el.dicomViewport);
            const deltaX = e.clientX - lastMousePos.x;
            const deltaY = e.clientY - lastMousePos.y;
            viewport.translation.x += deltaX / viewport.scale;
            viewport.translation.y += deltaY / viewport.scale;
            cornerstone.setViewport(el.dicomViewport, viewport);
            lastMousePos = { x: e.clientX, y: e.clientY };
        }
    });

    // Downloads
    el.downloadMaskBtn.addEventListener('click', () => {
        if (!currentPatientId) return;
        window.open(`${API_BASE_URL}/download/mask/${currentPatientId}`, '_blank');
    });

    el.downloadReportBtn.addEventListener('click', () => {
        if (!currentPatientId) return;
        // Text blob generation for simplicity over generating a new API endpoint if it wasn't specified strictly
        const textContent = `
LungSeg AI - Diagnostics Report
================================
Patient ID: ${el.v_patientId.textContent}
Date: ${new Date().toLocaleString()}

Detected Nodules: ${el.v_nodules.textContent}
Tumor Volume: ${el.v_volume.textContent}
Max Diameter: ${el.v_diameter.textContent}
Slice Range: ${el.v_range.textContent}
AI Confidence: ${el.v_confidenceVal.textContent}
        `;
        const blob = new Blob([textContent], { type: "text/plain;charset=utf-8" });
        const tempLink = document.createElement("a");
        tempLink.href = URL.createObjectURL(blob);
        tempLink.setAttribute("download", `${currentPatientId}_Report.txt`);
        tempLink.click();
    });
}

function updateWindowingDisplay(viewport) {
    el.wwDisplay.textContent = Math.round(viewport.voi.windowWidth);
    el.wlDisplay.textContent = Math.round(viewport.voi.windowCenter);
}

async function renderSlice(index) {
    if (index < 0 || index >= sliceDataList.length) return;
    currentSliceIndex = index;

    // Update UI controls
    el.sliceScrubber.value = index + 1;
    el.sliceIndicator.textContent = `Slice ${index + 1} / ${sliceDataList.length}`;

    // Update thumbnails highlighting
    document.querySelectorAll('.thumbnail-item').forEach((item, i) => {
        item.classList.toggle('active', i === index);
        if (i === index) item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });

    try {
        const image = await cornerstone.loadImage(sliceDataList[index]);
        cornerstone.displayImage(el.dicomViewport, image);
        // Mask overlay is triggered automatically by the 'cornerstoneimagerendered' event hook configured earlier.
    } catch (e) {
        console.error("Failed to render slice", index, e);
    }
}

function drawMaskOverlay() {
    const canvas = el.maskCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!isMaskVisible || !maskDataArray || !maskDataArray[currentSliceIndex] || maskDataArray[currentSliceIndex].length === 0) return;

    const sliceMask = maskDataArray[currentSliceIndex]; // Array of 0s and 1s corresponding to pixels

    // Obtain the image dimensions from the viewport to calculate mapping
    const image = cornerstone.getImage(el.dicomViewport);
    if (!image) return;

    // We assume the mask array length == image.width * image.height.
    // We create an ImageData object matching the *source image* size,
    // manipulate the pixels, create a temporary canvas with it, 
    // and then draw the temp canvas scaled onto the viewer canvas.
    // Note: Cornerstone handles scaling and translation, so drawing directly to the viewport canvas requires transforms matching cornerstone's render math.
    // Real implementation: We modify the cornerstone canvas transform.

    // A simpler HTML5 overlay approach matching Cornerstone transforms:
    const viewport = cornerstone.getViewport(el.dicomViewport);
    const scale = viewport.scale;

    ctx.save();

    // Apply center transform (cornerstone renders from center)
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.translate(viewport.translation.x * scale, viewport.translation.y * scale);
    ctx.scale(scale, scale);

    if (viewport.rotation !== 0) ctx.rotate((viewport.rotation * Math.PI) / 180);

    // Position back to top-left of the image domain
    ctx.translate(-image.width / 2, -image.height / 2);

    // Create ImageData for the mask
    const imgData = new ImageData(image.width, image.height);
    const data = imgData.data;

    // Populate pixels based on the flat array
    const alpha = Math.round(maskOpacity * 255);
    for (let i = 0; i < sliceMask.length; i++) {
        if (sliceMask[i] === 1 || sliceMask[i] > 0) { // boolean mask
            const idx = i * 4;
            data[idx] = 255;  // R
            data[idx + 1] = 70;   // G
            data[idx + 2] = 70;   // B
            data[idx + 3] = alpha; // A
        }
    }

    // Since we cannot draw ImageData scaled directly, we put it on a temporary canvas
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = image.width;
    tmpCanvas.height = image.height;
    tmpCanvas.getContext('2d').putImageData(imgData, 0, 0);

    // Draw the tmpCanvas onto our transformed context
    ctx.drawImage(tmpCanvas, 0, 0);

    ctx.restore();
}

async function generateThumbnails() {
    el.thumbnailContainer.innerHTML = ''; // Clear existing

    // Displaying 10-20 thumbnails max helps performance if there are 500 slices.
    // Requirements stated "Vertical list of slice thumbnails", we will create all but use IntersectionObserver if possible, or just append them and let DOM handle.
    for (let i = 0; i < sliceDataList.length; i++) {
        const wrap = document.createElement('div');
        wrap.className = `thumbnail-item ${i === currentSliceIndex ? 'active' : ''}`;

        const canvas = document.createElement('canvas');
        canvas.className = 'thumbnail-canvas';

        const label = document.createElement('div');
        label.className = 'thumbnail-label';
        label.textContent = `Slice ${i + 1}`;

        wrap.appendChild(canvas);
        wrap.appendChild(label);

        // Render indicator if mask has tumor pixels in this slice (simplified logic)
        let hasTumor = false;
        if (maskDataArray && maskDataArray[i]) {
            hasTumor = maskDataArray[i].some(v => v === 1 || v > 0);
        }

        if (hasTumor) {
            const indicator = document.createElement('div');
            indicator.className = 'thumbnail-indicator';
            indicator.title = 'Tumor Detected';
            wrap.appendChild(indicator);
        }

        wrap.addEventListener('click', () => renderSlice(i));
        el.thumbnailContainer.appendChild(wrap);

        // Queue thumbnail rendering using a tiny offscreen cornerstone element or just drawing it specifically
        // For performance, doing it asynchronously
        setTimeout(() => {
            renderThumbnail(sliceDataList[i], canvas);
        }, i * 50); // Stagger loading
    }
}

async function renderThumbnail(imageId, canvas) {
    try {
        cornerstone.enable(canvas);
        const image = await cornerstone.loadImage(imageId);
        cornerstone.displayImage(canvas, image);
        const vp = cornerstone.getViewport(canvas);
        vp.scale = canvas.clientWidth / image.width; // fit
        cornerstone.setViewport(canvas, vp);
    } catch (e) { /* ignore single thumbnail fail */ }
}

// ==========================================
// LOCAL STORAGE HISTORY
// ==========================================

function loadRunHistory() {
    let history = JSON.parse(localStorage.getItem('lungseg_history') || '[]');
    el.historyTbody.innerHTML = '';

    if (history.length === 0) {
        el.historyTbody.innerHTML = `<tr><td colspan="2" style="text-align:center;color:#9CA3AF">No runs yet</td></tr>`;
        return;
    }

    // Render history records
    history.slice().reverse().forEach((run, i) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${run.patientId}</td>
            <td>${run.date}</td>
        `;
        tr.title = `Nodules: ${run.nodules} | Vol: ${run.volume}`;

        // Click to mock "reload" (actually we'd fetch GET /results/:id and skip upload)
        tr.addEventListener('click', () => {
            currentPatientId = run.patientId;
            showToast(`Loading previous run: ${run.patientId}`);
            // If backend is active and holds data, loadResults(). 
            // Since we don't have the original DICOM files cached locally, viewer will fail.
            // A real SPA would request the server for the dicom series via WADO-URI or require re-upload.
            // loadResults(run.patientId);
        });

        el.historyTbody.appendChild(tr);
    });
}

function saveRunHistory(runData) {
    let history = JSON.parse(localStorage.getItem('lungseg_history') || '[]');
    // Avoid exact duplicates (same patient basically overrides unless we want multiple specific dates)
    history = history.filter(r => r.patientId !== runData.patientId);
    history.push(runData);
    if (history.length > 20) history.shift(); // keep last 20
    localStorage.setItem('lungseg_history', JSON.stringify(history));
    loadRunHistory();
}
