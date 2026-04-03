/**
 * CCTV Face Reconstruction System - Frontend Logic
 * Handles upload, enhancement, comparison, history, and video processing.
 */

// ============================================================
// State Management
// ============================================================
const state = {
    selectedModel: 'gfpgan',
    upscaleFactor: 2,
    frameInterval: 5,
    maxFrames: 50,
    currentFile: null,
    currentVideoFile: null,
    isProcessing: false,
    currentTab: 'enhance'
};

// ============================================================
// DOM Elements
// ============================================================
const $ = (id) => document.getElementById(id);
const $$ = (selector) => document.querySelectorAll(selector);

// ============================================================
// Initialization
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initImageUpload();
    initVideoUpload();
    initModelSelector();
    initUpscaleSelector();
    initFrameIntervalSelector();
    initCompareSlider();
    initEnhanceButton();
    initVideoEnhanceButton();
    loadStats();
    loadHistory();
});

// ============================================================
// Navigation
// ============================================================
function initNavigation() {
    $$('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            switchTab(tab);
        });
    });
}

function switchTab(tabName) {
    state.currentTab = tabName;

    // Update nav buttons
    $$('.nav-btn').forEach(b => b.classList.remove('active'));
    const activeBtn = document.querySelector(`.nav-btn[data-tab="${tabName}"]`);
    if (activeBtn) activeBtn.classList.add('active');

    // Update tab content
    $$('.tab-content').forEach(t => t.classList.remove('active'));
    const activeTab = $(`tab-${tabName}`);
    if (activeTab) activeTab.classList.add('active');

    // Refresh data for specific tabs
    if (tabName === 'history') loadHistory();
    if (tabName === 'stats') loadStats();
}

// ============================================================
// Image Upload
// ============================================================
function initImageUpload() {
    const zone = $('upload-zone');
    const input = $('file-input');

    // Click to browse
    zone.addEventListener('click', () => input.click());

    // Drag events
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });

    // File input change
    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageFile(e.target.files[0]);
        }
    });

    // Clear button
    $('clear-upload').addEventListener('click', (e) => {
        e.stopPropagation();
        clearImageUpload();
    });

    // New enhancement button
    $('new-enhance-btn').addEventListener('click', () => {
        clearImageUpload();
        $('results-card').classList.add('hidden');
        $('faces-card').classList.add('hidden');
    });
}

function handleImageFile(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    const allowed = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'];

    if (!allowed.includes(ext)) {
        showToast('Unsupported file format. Use: JPG, PNG, BMP, WebP', 'error');
        return;
    }

    if (file.size > 100 * 1024 * 1024) {
        showToast('File too large. Maximum size is 100MB.', 'error');
        return;
    }

    state.currentFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        $('preview-image').src = e.target.result;
        $('preview-container').classList.remove('hidden');
        $('upload-zone').style.display = 'none';

        // Show file info
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        $('preview-info').textContent = `${file.name} • ${sizeMB} MB`;
    };
    reader.readAsDataURL(file);

    // Enable enhance button
    $('enhance-btn').disabled = false;
    showToast('Image loaded successfully!', 'success');
}

function clearImageUpload() {
    state.currentFile = null;
    $('preview-container').classList.add('hidden');
    $('upload-zone').style.display = '';
    $('preview-image').src = '';
    $('file-input').value = '';
    $('enhance-btn').disabled = true;
}

// ============================================================
// Video Upload
// ============================================================
function initVideoUpload() {
    const zone = $('video-upload-zone');
    const input = $('video-file-input');

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleVideoFile(e.dataTransfer.files[0]);
        }
    });

    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleVideoFile(e.target.files[0]);
        }
    });

    $('clear-video').addEventListener('click', (e) => {
        e.stopPropagation();
        clearVideoUpload();
    });
}

function handleVideoFile(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    const allowed = ['.mp4', '.avi', '.mov', '.mkv', '.wmv'];

    if (!allowed.includes(ext)) {
        showToast('Unsupported video format. Use: MP4, AVI, MOV, MKV', 'error');
        return;
    }

    state.currentVideoFile = file;

    // Show preview
    const url = URL.createObjectURL(file);
    $('video-preview').src = url;
    $('video-preview-container').classList.remove('hidden');
    $('video-upload-zone').style.display = 'none';

    $('enhance-video-btn').disabled = false;
    showToast('Video loaded successfully!', 'success');
}

function clearVideoUpload() {
    state.currentVideoFile = null;
    $('video-preview-container').classList.add('hidden');
    $('video-upload-zone').style.display = '';
    $('video-preview').src = '';
    $('video-file-input').value = '';
    $('enhance-video-btn').disabled = true;
}

// ============================================================
// Model & Settings Selectors
// ============================================================
function initModelSelector() {
    $$('.model-option').forEach(opt => {
        opt.addEventListener('click', () => {
            $$('.model-option').forEach(o => o.classList.remove('active'));
            opt.classList.add('active');
            state.selectedModel = opt.dataset.model;
        });
    });
}

function initUpscaleSelector() {
    $$('#upscale-selector .toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('#upscale-selector .toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.upscaleFactor = parseInt(btn.dataset.value);
        });
    });
}

function initFrameIntervalSelector() {
    $$('#frame-interval-selector .toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('#frame-interval-selector .toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.frameInterval = parseInt(btn.dataset.value);
        });
    });
}

// ============================================================
// Image Enhancement
// ============================================================
function initEnhanceButton() {
    $('enhance-btn').addEventListener('click', enhanceImage);
}

async function enhanceImage() {
    if (!state.currentFile || state.isProcessing) return;

    state.isProcessing = true;
    const processingCard = $('processing-card');
    const resultsCard = $('results-card');
    const facesCard = $('faces-card');

    // Show processing
    processingCard.classList.remove('hidden');
    resultsCard.classList.add('hidden');
    facesCard.classList.add('hidden');
    $('enhance-btn').disabled = true;

    // Simulate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + Math.random() * 3, 90);
        $('progress-fill').style.width = progress + '%';

        if (progress < 20) {
            $('processing-status').textContent = 'Loading AI models...';
        } else if (progress < 50) {
            $('processing-status').textContent = 'Detecting faces...';
        } else if (progress < 80) {
            $('processing-status').textContent = 'Enhancing image with ' + state.selectedModel.toUpperCase() + '...';
        } else {
            $('processing-status').textContent = 'Finalizing results...';
        }
    }, 500);

    try {
        // Build form data
        const formData = new FormData();
        formData.append('file', state.currentFile);
        formData.append('model', state.selectedModel);
        formData.append('upscale', state.upscaleFactor);
        formData.append('face_enhance', true);
        formData.append('bg_enhance', true);

        // Send request
        const response = await fetch('/api/enhance/image', {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Enhancement failed');
        }

        const result = await response.json();

        // Complete progress
        $('progress-fill').style.width = '100%';
        $('processing-status').textContent = 'Complete!';

        setTimeout(() => {
            processingCard.classList.add('hidden');
            displayResults(result);
        }, 500);

    } catch (error) {
        clearInterval(progressInterval);
        processingCard.classList.add('hidden');
        showToast('Enhancement failed: ' + error.message, 'error');
        $('enhance-btn').disabled = false;
    }

    state.isProcessing = false;
}

function displayResults(result) {
    const resultsCard = $('results-card');
    const facesCard = $('faces-card');

    // Update stats
    $('result-stats').textContent =
        `${result.faces_detected} faces • ${result.processing_time}s • ${result.model_used.toUpperCase()} ${result.upscale_factor}x`;

    // Set comparison images
    $('compare-before-img').src = result.original_url;
    $('compare-after-img').src = result.enhanced_url;

    // Download button
    $('download-btn').href = result.enhanced_url;
    $('download-btn').download = 'enhanced_' + result.original_filename;

    // Show results
    resultsCard.classList.remove('hidden');

    // Display detected faces
    if (result.faces && result.faces.length > 0) {
        $('face-count').textContent = result.faces.length;
        const grid = $('faces-grid');
        grid.innerHTML = '';

        result.faces.forEach((face, i) => {
            const card = document.createElement('div');
            card.className = 'face-card';
            card.innerHTML = `
                <div class="face-card-images">
                    <img src="${face.cropped_url}" alt="Original Face ${i+1}" loading="lazy">
                    <img src="${face.enhanced_url}" alt="Enhanced Face ${i+1}" loading="lazy">
                </div>
                <div class="face-card-label">
                    <span>Face ${i+1}</span>
                    <span class="confidence">${(face.confidence * 100).toFixed(0)}%</span>
                </div>
            `;
            grid.appendChild(card);
        });

        facesCard.classList.remove('hidden');
    }

    // Reset compare slider position
    resetCompareSlider();

    showToast(`Image enhanced successfully! ${result.faces_detected} face(s) detected.`, 'success');
}

// ============================================================
// Before/After Comparison Slider
// ============================================================
function initCompareSlider() {
    const wrapper = document.querySelector('.compare-wrapper');
    if (!wrapper) return;

    let isDragging = false;

    const updateSlider = (x) => {
        const rect = wrapper.getBoundingClientRect();
        let pos = ((x - rect.left) / rect.width) * 100;
        pos = Math.max(0, Math.min(100, pos));

        $('compare-slider').style.left = pos + '%';
        $('compare-after').style.clipPath = `inset(0 0 0 ${pos}%)`;
    };

    wrapper.addEventListener('mousedown', (e) => {
        isDragging = true;
        updateSlider(e.clientX);
    });

    document.addEventListener('mousemove', (e) => {
        if (isDragging) {
            e.preventDefault();
            updateSlider(e.clientX);
        }
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
    });

    // Touch support
    wrapper.addEventListener('touchstart', (e) => {
        isDragging = true;
        updateSlider(e.touches[0].clientX);
    });

    wrapper.addEventListener('touchmove', (e) => {
        if (isDragging) {
            updateSlider(e.touches[0].clientX);
        }
    });

    wrapper.addEventListener('touchend', () => {
        isDragging = false;
    });
}

function resetCompareSlider() {
    $('compare-slider').style.left = '50%';
    $('compare-after').style.clipPath = 'inset(0 0 0 50%)';
}

// ============================================================
// Video Enhancement
// ============================================================
function initVideoEnhanceButton() {
    $('enhance-video-btn').addEventListener('click', enhanceVideo);
}

async function enhanceVideo() {
    if (!state.currentVideoFile || state.isProcessing) return;

    state.isProcessing = true;
    const processingCard = $('video-processing-card');
    const resultsCard = $('video-results-card');

    processingCard.classList.remove('hidden');
    resultsCard.classList.add('hidden');
    $('enhance-video-btn').disabled = true;

    // Progress simulation
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + Math.random() * 2, 90);
        $('video-progress-fill').style.width = progress + '%';
        $('video-processing-status').textContent =
            `Processing frames... (${Math.floor(progress)}%)`;
    }, 1000);

    try {
        const maxFrames = parseInt($('max-frames').value) || 50;

        const formData = new FormData();
        formData.append('file', state.currentVideoFile);
        formData.append('model', state.selectedModel);
        formData.append('upscale', state.upscaleFactor);
        formData.append('frame_interval', state.frameInterval);
        formData.append('max_frames', maxFrames);

        const response = await fetch('/api/enhance/video', {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Video enhancement failed');
        }

        const result = await response.json();

        $('video-progress-fill').style.width = '100%';

        setTimeout(() => {
            processingCard.classList.add('hidden');
            displayVideoResults(result);
        }, 500);

    } catch (error) {
        clearInterval(progressInterval);
        processingCard.classList.add('hidden');
        showToast('Video enhancement failed: ' + error.message, 'error');
        $('enhance-video-btn').disabled = false;
    }

    state.isProcessing = false;
}

function displayVideoResults(result) {
    const content = $('video-results-content');
    content.innerHTML = `
        <video controls>
            <source src="${result.enhanced_url}" type="video/mp4">
            Your browser does not support video playback.
        </video>
        <div class="video-stats">
            <div class="video-stat">
                <div class="video-stat-value">${result.frames_processed}</div>
                <div class="video-stat-label">Frames Processed</div>
            </div>
            <div class="video-stat">
                <div class="video-stat-value">${result.total_faces_detected}</div>
                <div class="video-stat-label">Faces Detected</div>
            </div>
            <div class="video-stat">
                <div class="video-stat-value">${result.processing_time}s</div>
                <div class="video-stat-label">Processing Time</div>
            </div>
        </div>
        <div class="result-actions">
            <a class="btn btn-primary" href="${result.enhanced_url}" download>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                Download Enhanced Video
            </a>
        </div>
    `;

    $('video-results-card').classList.remove('hidden');
    showToast(`Video enhanced! ${result.frames_processed} frames processed.`, 'success');
}

// ============================================================
// History
// ============================================================
async function loadHistory() {
    try {
        const status = $('history-filter-status')?.value || '';
        const type = $('history-filter-type')?.value || '';

        let url = '/api/jobs?limit=50';
        if (status) url += `&status=${status}`;
        if (type) url += `&file_type=${type}`;

        const response = await fetch(url);
        if (!response.ok) return;

        const data = await response.json();
        displayHistory(data.jobs);
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

function displayHistory(jobs) {
    const list = $('history-list');

    if (!jobs || jobs.length === 0) {
        list.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polyline points="12,6 12,12 16,14"/></svg>
                <p>No processing history yet</p>
                <p class="text-muted">Enhanced images and videos will appear here</p>
            </div>
        `;
        return;
    }

    list.innerHTML = jobs.map(job => {
        const date = new Date(job.created_at).toLocaleString();
        const statusClass = `status-${job.status}`;
        const thumb = job.original_url || '';
        const time = job.processing_time ? `${job.processing_time}s` : '-';

        return `
            <div class="history-item">
                <img class="history-thumb" src="${thumb}" alt="${job.original_filename}"
                     onerror="this.style.display='none'">
                <div class="history-info">
                    <div class="history-filename">${job.original_filename}</div>
                    <div class="history-meta">
                        <span>${job.model_used.toUpperCase()} ${job.upscale_factor}x</span>
                        <span>${job.faces_detected} faces</span>
                        <span>${time}</span>
                        <span>${date}</span>
                    </div>
                </div>
                <span class="history-status ${statusClass}">${job.status}</span>
                <div class="history-actions">
                    ${job.enhanced_url ? `
                        <a class="btn btn-ghost btn-sm" href="${job.enhanced_url}" download title="Download">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                        </a>
                    ` : ''}
                    <button class="btn btn-danger btn-sm" onclick="deleteJob('${job.id}')" title="Delete">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3,6 5,6 21,6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

// History filters
document.addEventListener('DOMContentLoaded', () => {
    const statusFilter = $('history-filter-status');
    const typeFilter = $('history-filter-type');

    if (statusFilter) statusFilter.addEventListener('change', loadHistory);
    if (typeFilter) typeFilter.addEventListener('change', loadHistory);
});

async function deleteJob(jobId) {
    if (!confirm('Are you sure you want to delete this job?')) return;

    try {
        const response = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
        if (response.ok) {
            showToast('Job deleted successfully', 'success');
            loadHistory();
            loadStats();
        } else {
            showToast('Failed to delete job', 'error');
        }
    } catch (error) {
        showToast('Error: ' + error.message, 'error');
    }
}

// ============================================================
// Stats
// ============================================================
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) return;

        const stats = await response.json();

        $('stat-total-jobs').textContent = stats.total_jobs;
        $('stat-completed').textContent = stats.completed_jobs;
        $('stat-faces').textContent = stats.total_faces_detected;
        $('stat-avg-time').textContent = stats.avg_processing_time ? stats.avg_processing_time + 's' : '0s';

        // System info
        if (stats.models_info) {
            $('info-device').textContent = stats.models_info.device.toUpperCase();
            $('info-gfpgan').textContent = stats.models_info.gfpgan_loaded ? '✅ Loaded' : '⏳ Not loaded';
            $('info-realesrgan').textContent = stats.models_info.bg_upsampler_loaded ? '✅ Loaded' : '⏳ Not loaded';
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// ============================================================
// Toast Notifications
// ============================================================
function showToast(message, type = 'info') {
    const container = $('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icons = {
        success: '✓',
        error: '✕',
        info: 'ℹ'
    };

    toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span> ${message}`;
    container.appendChild(toast);

    // Auto-remove after 5s
    setTimeout(() => {
        if (toast.parentNode) {
            toast.remove();
        }
    }, 5000);
}
