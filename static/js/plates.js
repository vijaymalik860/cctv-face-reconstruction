/**
 * plates.js — Number Plate Detection Tab
 * Handles all UI logic for the "Number Plate" tab:
 *  - File upload (image & video)
 *  - API call to /api/plates/detect or /api/plates/detect/video
 *  - Pipeline progress animation (4 steps)
 *  - Results rendering (annotated image + per-plate cards)
 *  - History loading
 *  - Night-vision toggle + copy-to-clipboard
 */

(function () {
  "use strict";

  /* ─── State ─────────────────────────────────────────────── */
  let plateFile   = null;
  let plateUpscale = 4;
  let forceNight  = false;
  let isVideo     = false;
  let plateFrameInterval = 10;
  let plateConf = 0.25;
  let plateOcr = 0.0;

  /* ─── DOM Refs ───────────────────────────────────────────── */
  const uploadZone     = document.getElementById("plate-upload-zone");
  const fileInput      = document.getElementById("plate-file-input");
  const previewCont    = document.getElementById("plate-preview-container");
  const previewImg     = document.getElementById("plate-preview-image");
  const previewVid     = document.getElementById("plate-preview-video");
  const clearBtn       = document.getElementById("plate-clear-upload");
  const detectBtn      = document.getElementById("plate-detect-btn");
  const nightToggle    = document.getElementById("plate-night-toggle");
  const nightLabel     = document.getElementById("plate-night-label");
  const processingCard = document.getElementById("plate-processing-card");
  const resultsCard    = document.getElementById("plate-results-card");
  const detailsCard    = document.getElementById("plate-details-card");
  const emptyCard      = document.getElementById("plate-empty-card");
  const progressFill   = document.getElementById("plate-progress-fill");
  const statusText     = document.getElementById("plate-processing-status");
  const resultsGrid    = document.getElementById("plate-results-grid");
  const countBadge     = document.getElementById("plate-count-badge");
  const historyList    = document.getElementById("plate-history-list");
  
  // Compare Feature Nodes
  const compareContainer = document.getElementById("plate-compare-container");
  const compareBeforeImg = document.getElementById("plate-compare-before-img");
  const compareAfterImg  = document.getElementById("plate-compare-after-img");
  const compareAfter     = document.getElementById("plate-compare-after");
  const compareSlider    = document.getElementById("plate-compare-slider");
  const downloadBtn      = document.getElementById("plate-download-btn");
  const newBtn           = document.getElementById("plate-new-btn");
  
  /* ─── Compare Slider Logic ─────────────────────────────── */
  let isDragging = false;
  if(compareSlider) {
      compareSlider.addEventListener('mousedown', () => isDragging = true);
      window.addEventListener('mouseup', () => isDragging = false);
      window.addEventListener('mousemove', (e) => {
          if (!isDragging) return;
          const rect = compareContainer.getBoundingClientRect();
          let x = e.clientX - rect.left;
          x = Math.max(0, Math.min(x, rect.width));
          const percent = (x / rect.width) * 100;
          compareAfter.style.width = `${percent}%`;
          compareSlider.style.left = `${percent}%`;
      });
  }
  const histRefreshBtn = document.getElementById("plate-history-refresh");
  const videoSettings  = document.getElementById("plate-video-settings");
  const plateMaxFrames = document.getElementById("plate-max-frames");

  /* ─── Upscale Toggle ──────────────────────────────────── */
  document.getElementById("plate-upscale-selector")
    .querySelectorAll(".toggle-btn")
    .forEach((btn) => {
      btn.addEventListener("click", () => {
        document.getElementById("plate-upscale-selector")
          .querySelectorAll(".toggle-btn")
          .forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        plateUpscale = parseInt(btn.dataset.value, 10);
      });
    });

  /* ─── Frame Interval Toggle ───────────────────────────── */
  document.getElementById("plate-frame-interval-selector")
    ?.querySelectorAll(".toggle-btn")
    .forEach((btn) => {
      btn.addEventListener("click", () => {
        document.getElementById("plate-frame-interval-selector")
          .querySelectorAll(".toggle-btn")
          .forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        plateFrameInterval = parseInt(btn.dataset.value, 10);
      });
    });

  /* ─── Night-vision Toggle ─────────────────────────────── */
  nightToggle.addEventListener("change", () => {
    forceNight = nightToggle.checked;
    nightLabel.textContent = forceNight
      ? "Force Night-Vision ON"
      : "Auto-detect (recommended)";
    nightLabel.style.color = forceNight ? "#00e676" : "";
    // Visual feedback on the toggle slider
    nightToggle.nextElementSibling.style.background = forceNight
      ? "#00e676"
      : "#2a2a3a";
  });

  /* ─── Threshold Sliders ──────────────────────────────── */
  const confSlider = document.getElementById("plate-conf-slider");
  const confVal    = document.getElementById("plate-conf-val");
  if (confSlider) {
      confSlider.addEventListener("input", (e) => {
          plateConf = parseFloat(e.target.value);
          if (confVal) confVal.textContent = Math.round(plateConf * 100) + "%";
      });
  }

  const ocrSlider = document.getElementById("plate-ocr-slider");
  const ocrVal    = document.getElementById("plate-ocr-val");
  if (ocrSlider) {
      ocrSlider.addEventListener("input", (e) => {
          plateOcr = parseFloat(e.target.value);
          if (ocrVal) ocrVal.textContent = Math.round(plateOcr * 100) + "%";
      });
  }

  /* ─── Upload Zone ────────────────────────────────────────── */
  uploadZone.addEventListener("click", () => fileInput.click());

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });
  uploadZone.addEventListener("dragleave", () =>
    uploadZone.classList.remove("drag-over")
  );
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  clearBtn.addEventListener("click", resetUpload);

  function handleFile(f) {
    plateFile = f;
    isVideo   = f.type.startsWith("video/");

    previewCont.classList.remove("hidden");
    uploadZone.style.display = "none";

    if (isVideo) {
      previewImg.style.display = "none";
      previewVid.style.display = "block";
      previewVid.src = URL.createObjectURL(f);
      if (videoSettings) show(videoSettings);
    } else {
      previewVid.style.display = "none";
      previewImg.style.display = "block";
      previewImg.src = URL.createObjectURL(f);
    }

    detectBtn.disabled = false;
  }

  function resetUpload() {
    plateFile = null;
    fileInput.value = "";
    previewCont.classList.add("hidden");
    uploadZone.style.display = "";
    previewImg.src = "";
    previewVid.src = "";
    detectBtn.disabled = true;
    if (videoSettings) hide(videoSettings);

    hide(processingCard);
    hide(resultsCard);
    hide(detailsCard);
    show(emptyCard);
  }

  /* ─── Detect Button ──────────────────────────────────── */
  detectBtn.addEventListener("click", runDetection);

  async function runDetection() {
    if (!plateFile) return;

    // UI transitions
    detectBtn.disabled = true;
    hide(resultsCard);
    hide(emptyCard);
    show(processingCard);

    // Animate pipeline steps
    resetPipelineSteps();
    animateProgress(0, 20, 800);
    updateStatus("🔍 Step 1: YOLOv8 — detecting number plates...");
    await delay(500);
    activatePipelineStep("pstep-detect");

    try {
      const formData = new FormData();
      formData.append("file",    plateFile);
      formData.append("upscale", plateUpscale);
      formData.append("force_night", forceNight);
      formData.append("conf_threshold", plateConf);
      formData.append("ocr_threshold", plateOcr);
      
      if (isVideo) {
        formData.append("frame_interval", plateFrameInterval);
        formData.append("max_frames", plateMaxFrames.value || 30);
      }

      const endpoint = isVideo
        ? "/api/plates/detect/video"
        : "/api/plates/detect";

      // Fake step animations (real pipeline runs server-side, so
      // we animate the steps while awaiting the API response)
      const stepTimer = fakeStepAnimation();

      const res  = await fetch(endpoint, { method: "POST", body: formData });
      const data = await res.json();

      clearInterval(stepTimer);

      if (!res.ok) {
        throw new Error(data.detail || "Detection failed");
      }

      // Complete all steps
      ["pstep-detect","pstep-rectify","pstep-enhance","pstep-ocr"].forEach(activatePipelineStep);
      animateProgress(100, 100, 300);
      updateStatus("✅ Pipeline complete!");

      await delay(600);
      renderResults(data);
      loadHistory();

    } catch (err) {
      hide(processingCard);
      show(emptyCard);
      showToast("error", "Detection Failed", err.message);
      detectBtn.disabled = false;
    }
  }

  /* Simulate the 4-step animation while waiting for the server */
  function fakeStepAnimation() {
    const steps  = ["pstep-detect","pstep-rectify","pstep-enhance","pstep-ocr"];
    const labels = [
      "🔍 Step 1: YOLOv8 — detecting plates...",
      "📐 Step 2: WPOD-NET — rectifying perspective...",
      "✨ Step 3: Real-ESRGAN ×4 — sharpening...",
      "📝 Step 4: PaddleOCR — reading text...",
    ];
    const pcts  = [25, 50, 75, 90];
    let i = 0;

    const timer = setInterval(() => {
      if (i < steps.length) {
        activatePipelineStep(steps[i]);
        updateStatus(labels[i]);
        animateProgress(pcts[i], pcts[i], 400);
        i++;
      }
    }, 2200);

    return timer;
  }

  /* ─── Render Results ──────────────────────────────────── */
  function renderResults(data) {
    hide(processingCard);
    show(resultsCard);
    
    // Details card handles the grid and the list (we extracted grid from resultsCard previously, wait I did that in index.html)
    if(detailsCard) show(detailsCard);
    hide(emptyCard);

    const plates = data.plates || [];
    countBadge.textContent = `${plates.length} plate${plates.length !== 1 ? "s" : ""}`;

    // Comparison slider images
    if (data.original_url && data.annotated_url) {
      compareBeforeImg.src = data.original_url + "?t=" + Date.now();
      compareAfterImg.src = data.annotated_url + "?t=" + Date.now();
      compareAfterImg.style.display = "block";
      
      // Reset slider to middle
      compareAfter.style.width = '50%';
      compareSlider.style.left = '50%';
      
      // Handle download button
      if (downloadBtn) Object.assign(downloadBtn, { href: data.annotated_url });
    } else {
      if(compareBeforeImg) compareBeforeImg.src = "";
      if(compareAfterImg) compareAfterImg.src = "";
    }

    // Per-plate cards
    resultsGrid.innerHTML = "";
    if (plates.length === 0) {
      resultsGrid.innerHTML = `
        <div style="text-align:center;padding:24px;color:rgba(255,255,255,.4);">
          No number plates detected. Try uploading a clearer image or enabling Night Vision mode.
        </div>`;
    } else {
      plates.forEach((plate, i) => {
        resultsGrid.appendChild(buildPlateCard(plate, i));
      });
    }

    detectBtn.disabled = false;
    showToast("success", "Detection Complete",
      `${plates.length} plate(s) found in ${(data.processing_time||0).toFixed(1)}s`);
  }

  function buildPlateCard(plate, i) {
    const night  = plate.is_night_vision;
    const conf   = Math.round((plate.detection_confidence || 0) * 100);
    const ocrConf = Math.round((plate.ocr_confidence || 0) * 100);
    const text   = plate.ocr_text || "(no text detected)";

    const card = document.createElement("div");
    card.style.cssText = `
      background:rgba(255,255,255,.04);
      border:1px solid rgba(255,255,255,.09);
      border-radius:12px;
      padding:16px;
      transition:border-color .2s;
    `;
    card.onmouseenter = () => card.style.borderColor = "rgba(0,230,118,.35)";
    card.onmouseleave = () => card.style.borderColor = "rgba(255,255,255,.09)";

    card.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <span style="font-weight:600;color:#fff;font-size:.93rem;">
          Plate #${i + 1}
          ${night ? '<span style="font-size:.72rem;background:rgba(0,206,209,.2);color:#00ced1;padding:2px 7px;border-radius:4px;margin-left:6px;">🌙 IR</span>' : ""}
        </span>
        <span style="font-size:.78rem;color:rgba(255,255,255,.45);">
          Detect: ${conf}%
        </span>
      </div>

      <!-- 3-stage image strip: raw / rectified / enhanced -->
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:14px;">
        <div style="text-align:center;">
          <img src="${plate.original_crop_url}?t=${Date.now()}"
               alt="Raw" style="width:100%;border-radius:6px;border:1px solid rgba(255,255,255,.08);cursor:pointer;"
               onclick="openModal('Raw Crop — Plate #${i+1}','<img src=\\'${plate.original_crop_url}\\' style=\\'max-width:100%;border-radius:8px;\\'>')">
          <p style="font-size:.7rem;color:rgba(255,255,255,.35);margin:4px 0 0;">Raw Crop</p>
        </div>
        <div style="text-align:center;">
          <img src="${plate.rectified_url}?t=${Date.now()}"
               alt="Rectified" style="width:100%;border-radius:6px;border:1px solid rgba(255,255,255,.08);cursor:pointer;"
               onclick="openModal('Rectified Plate — Plate #${i+1}','<img src=\\'${plate.rectified_url}\\' style=\\'max-width:100%;border-radius:8px;\\'>')">
          <p style="font-size:.7rem;color:rgba(255,255,255,.35);margin:4px 0 0;">Rectified</p>
        </div>
        <div style="text-align:center;">
          <img src="${plate.enhanced_url}?t=${Date.now()}"
               alt="Enhanced" style="width:100%;border-radius:6px;border:1px solid rgba(0,230,118,.35);cursor:pointer;"
               onclick="openModal('Enhanced Plate — Plate #${i+1}','<img src=\\'${plate.enhanced_url}\\' style=\\'max-width:100%;border-radius:8px;\\'>')">
          <p style="font-size:.7rem;color:#00e676;margin:4px 0 0;">Enhanced ✨</p>
        </div>
      </div>

      <!-- OCR result banner -->
      <div style="background:rgba(0,0,0,.3);border:1px solid rgba(0,230,118,.25);border-radius:8px;padding:12px 14px;display:flex;align-items:center;justify-content:space-between;gap:10px;">
        <div>
          <p style="font-size:.72rem;color:rgba(255,255,255,.4);margin:0 0 3px;">OCR Text (${ocrConf}% confidence)</p>
          <p style="font-family:'JetBrains Mono',monospace;font-size:1.25rem;font-weight:700;color:#00e676;letter-spacing:.12em;margin:0;">
            ${text}
          </p>
        </div>
        <button onclick="copyText('${text.replace(/'/g,"\\'")}',this)"
                style="background:rgba(0,230,118,.12);border:1px solid rgba(0,230,118,.3);color:#00e676;border-radius:6px;padding:7px 12px;cursor:pointer;font-size:.78rem;white-space:nowrap;transition:all .2s;">
          📋 Copy
        </button>
      </div>
    `;
    return card;
  }

  /* ─── New Detection Button ───────────────────────────── */
  newBtn.addEventListener("click", resetUpload);

  /* ─── History ─────────────────────────────────────────── */
  histRefreshBtn.addEventListener("click", loadHistory);

  async function loadHistory() {
    try {
      const res  = await fetch("/api/plates/jobs?limit=10");
      const data = await res.json();
      renderHistory(data.jobs || []);
    } catch (e) {
      console.warn("[plates] History load failed:", e);
    }
  }

  function renderHistory(jobs) {
    if (!jobs.length) {
      historyList.innerHTML = `
        <div class="empty-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="6" width="20" height="12" rx="3"/><path d="M7 10h.01M17 10h.01M7 14h10"/></svg>
          <p>No plate detections yet</p>
          <p class="text-muted">Previous jobs will appear here</p>
        </div>`;
      return;
    }

    historyList.innerHTML = jobs.map((j) => {
      const statusColor = {
        completed: "#00e676", failed: "#ff4d6d", processing: "#ffd600"
      }[j.status] || "#aaa";

      const allTexts = (j.plates || [])
        .map((p) => p.ocr_text)
        .filter(Boolean)
        .join(", ") || "—";

      return `
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:14px 16px;border-bottom:1px solid rgba(255,255,255,.06);
                    gap:10px;flex-wrap:wrap;">
          <div style="flex:1;min-width:0;">
            <p style="margin:0 0 3px;font-weight:500;color:#fff;font-size:.88rem;
                      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
              ${j.original_filename}
            </p>
            <p style="margin:0;font-size:.76rem;color:rgba(255,255,255,.4);">
              ${j.plates_detected} plate(s) &nbsp;·&nbsp;
              ${(j.processing_time||0).toFixed(1)}s &nbsp;·&nbsp;
              ${j.created_at ? new Date(j.created_at).toLocaleString() : ""}
            </p>
            <p style="margin:3px 0 0;font-family:'JetBrains Mono',monospace;font-size:.8rem;color:#00e676;">
              ${allTexts}
            </p>
          </div>
          <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:.75rem;padding:3px 9px;border-radius:4px;
                         background:${statusColor}20;color:${statusColor};white-space:nowrap;">
              ${j.status}
            </span>
            <button onclick="downloadPlateJob('${j.id}', '${j.original_filename}')" title="Download CSV Report" style="background:none;border:none;color:#00e676;cursor:pointer;padding:4px;display:flex;align-items:center;">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            </button>
            <button onclick="deletePlateJob('${j.id}')" title="Delete Job" style="background:none;border:none;color:#ff4d6d;cursor:pointer;padding:4px;display:flex;align-items:center;">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>
            </button>
          </div>
        </div>`;
    }).join("");
  }

  /* ─── Pipeline Step Helpers ─────────────────────────── */
  function resetPipelineSteps() {
    ["pstep-detect","pstep-rectify","pstep-enhance","pstep-ocr"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        el.classList.remove("active");
        const dot = el.querySelector(".pipeline-dot");
        if (dot) dot.style.background = "rgba(255,255,255,.2)";
        el.style.color = "rgba(255,255,255,.4)";
      }
    });
    animateProgress(0, 0, 0);
  }

  function activatePipelineStep(id) {
    const el = document.getElementById(id);
    if (!el) return;
    const dot = el.querySelector(".pipeline-dot");
    if (dot) dot.style.background = "#00e676";
    el.style.color = "#00e676";
  }

  function animateProgress(from, to, ms) {
    if (!progressFill) return;
    progressFill.style.transition = ms ? `width ${ms}ms ease` : "none";
    progressFill.style.width = to + "%";
  }

  function updateStatus(msg) {
    if (statusText) statusText.textContent = msg;
  }

  /* ─── Modal ─────────────────────────────────────────── */
  window.openModal = function (title, contentHtml) {
    const modal = document.getElementById("plate-modal");
    document.getElementById("modal-title").textContent = title;
    document.getElementById("modal-content").innerHTML = contentHtml;
    modal.style.display = "flex";
  };

  /* ─── Copy to Clipboard ─────────────────────────────── */
  window.copyText = function (text, btn) {
    navigator.clipboard.writeText(text).then(() => {
      const orig = btn.textContent;
      btn.textContent = "✅ Copied!";
      btn.style.background = "rgba(0,230,118,.25)";
      setTimeout(() => {
        btn.textContent = orig;
        btn.style.background = "rgba(0,230,118,.12)";
      }, 1800);
    });
  };

  /* ─── History Item Actions ─────────────────────────── */
  window.downloadPlateJob = async function(jobId, filename) {
    try {
      const res = await fetch(`/api/plates/jobs/${jobId}`);
      if (!res.ok) throw new Error("Failed to fetch job details");
      const job = await res.json();
      
      let csv = "Plate Index,OCR Text,Confidence %,Night Vision,Original Image URL,Annotated Image URL\n";
      (job.plates || []).forEach((p, i) => {
        csv += `${i+1},"${p.ocr_text || ''}",${Math.round((p.ocr_confidence || 0)*100)},${p.is_night_vision ? 'Yes' : 'No'},"${window.location.origin}${job.original_url}","${job.annotated_url ? window.location.origin + job.annotated_url : ''}"\n`;
      });
      
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `plate_report_${filename}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      showToast("error", "Download Failed", e.message);
    }
  };

  window.deletePlateJob = async function(jobId) {
    if (!confirm("Are you sure you want to delete this detection job and all its files?")) return;
    try {
      const res = await fetch(`/api/plates/jobs/${jobId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error("Failed to delete job");
      showToast("success", "Deleted", "Job deleted successfully");
      loadHistory();
    } catch (e) {
      showToast("error", "Delete Failed", e.message);
    }
  };

  /* ─── Utility ────────────────────────────────────────── */
  function show(el) { el?.classList.remove("hidden"); }
  function hide(el) { el?.classList.add("hidden"); }
  function delay(ms) { return new Promise((r) => setTimeout(r, ms)); }

  function showToast(type, title, msg) {
    // Reuse existing app.js toast if available
    if (typeof window.showToast === "function") {
      window.showToast(type, title, msg);
      return;
    }
    console.log(`[${type}] ${title}: ${msg}`);
  }

  /* ─── Pipeline Step CSS injection ─────────────────── */
  const style = document.createElement("style");
  style.textContent = `
    .pipeline-step{
      display:flex;align-items:center;gap:8px;
      color:rgba(255,255,255,.4);font-size:.82rem;
      padding:8px 12px;border-radius:8px;
      border:1px solid rgba(255,255,255,.07);
      transition:all .3s;
    }
    .pipeline-dot{
      width:10px;height:10px;border-radius:50%;
      background:rgba(255,255,255,.2);
      flex-shrink:0;transition:background .3s;
    }
    #plate-upload-zone.drag-over{
      border-color:#00e676 !important;
      background:rgba(0,230,118,.06) !important;
    }
  `;
  document.head.appendChild(style);

  /* ─── Init ───────────────────────────────────────────── */
  loadHistory();

  // Also load history when the Plates tab becomes active
  document.getElementById("nav-plates")?.addEventListener("click", loadHistory);

})();
