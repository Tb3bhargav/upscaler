const SETTINGS_KEY = "ibpDcciSidebarSettings";
const DEFAULT_BRIDGE_PORT = "8765";
const HEALTH_TIMEOUT_MS = 5000;
const UPSCALE_TIMEOUT_MS = 5 * 60 * 1000;
const BRIDGE_START_COMMAND = ".venv\\Scripts\\python.exe bridge_server.py --host 127.0.0.1 --port 8765";

const refs = {
  backendUrl: document.getElementById("backendUrl"),
  checkConnection: document.getElementById("checkConnection"),
  imageInput: document.getElementById("imageInput"),
  dropZone: document.getElementById("dropZone"),
  queueSummary: document.getElementById("queueSummary"),
  resultLinks: document.getElementById("resultLinks"),
  scale: document.getElementById("scale"),
  runUpscale: document.getElementById("runUpscale"),
  clearQueue: document.getElementById("clearQueue"),
  statusText: document.getElementById("statusText"),
  upscaledPreview: document.getElementById("upscaledPreview"),
  downloadResult: document.getElementById("downloadResult")
};

const queue = [];
let nextJobId = 1;
let isProcessing = false;

function normalizeBridgeBaseFromInput(input) {
  const trimmed = String(input || "").trim();
  if (!trimmed) {
    throw new Error("Enter a local bridge URL first.");
  }

  let candidate = trimmed;
  if (!/^[a-zA-Z][a-zA-Z\d+\-.]*:\/\//.test(candidate)) {
    candidate = `http://${candidate}`;
  }

  let parsed;
  try {
    parsed = new URL(candidate);
  } catch {
    throw new Error("Bridge URL is invalid. Example: http://127.0.0.1:8765");
  }

  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("Bridge URL must use http or https.");
  }

  if (!parsed.hostname) {
    throw new Error("Bridge URL is missing a hostname.");
  }

  if (!parsed.port) {
    parsed.port = DEFAULT_BRIDGE_PORT;
  }

  return `${parsed.protocol}//${parsed.hostname}:${parsed.port}`;
}

function setStatus(message, kind = "info") {
  refs.statusText.textContent = message;
  refs.statusText.className = `status ${kind}`;
}

function parseIntegerInput(input, fallback) {
  const value = Number.parseInt(input.value, 10);
  return Number.isFinite(value) ? value : fallback;
}

function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  return fetch(url, {
    ...(options || {}),
    signal: controller.signal
  }).finally(() => {
    clearTimeout(timer);
  });
}

function buildBridgeCandidates() {
  const primary = normalizeBridgeBaseFromInput(refs.backendUrl.value);
  const parsed = new URL(primary);
  const candidates = [primary];

  if (parsed.hostname === "127.0.0.1") {
    candidates.push(`${parsed.protocol}//localhost:${parsed.port}`);
  } else if (parsed.hostname === "localhost") {
    candidates.push(`${parsed.protocol}//127.0.0.1:${parsed.port}`);
  }

  return [...new Set(candidates)];
}

function isFailedFetchError(error) {
  const message = String(error?.message || error || "");
  return message.toLowerCase().includes("failed to fetch") || message.toLowerCase().includes("networkerror");
}

function toBridgeError(error) {
  if (!error) {
    return `Bridge check failed. Start bridge with: ${BRIDGE_START_COMMAND}`;
  }

  if (error.name === "AbortError") {
    return `Bridge request timed out. Start bridge with: ${BRIDGE_START_COMMAND}`;
  }

  if (isFailedFetchError(error)) {
    return `Could not reach the bridge. Start bridge with: ${BRIDGE_START_COMMAND}`;
  }

  return String(error.message || error);
}

async function probeBridge(baseUrl) {
  const response = await fetchWithTimeout(
    `${baseUrl}/health`,
    { method: "GET" },
    HEALTH_TIMEOUT_MS
  );

  if (!response.ok) {
    throw new Error(`Bridge returned HTTP ${response.status}`);
  }

  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("Bridge health response was not valid JSON.");
  }

  if (!data.ok) {
    throw new Error("Bridge health response was not OK.");
  }

  return data;
}

async function resolveReachableBridge() {
  const candidates = buildBridgeCandidates();
  let lastError = null;

  for (const candidate of candidates) {
    try {
      await probeBridge(candidate);
      refs.backendUrl.value = candidate;
      return candidate;
    } catch (error) {
      lastError = error;
    }
  }

  throw new Error(toBridgeError(lastError));
}

async function postUpscale(baseUrl, payload) {
  const response = await fetchWithTimeout(
    `${baseUrl}/upscale`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    },
    UPSCALE_TIMEOUT_MS
  );

  let data;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    data = await response.json();
  } else {
    const text = await response.text();
    data = { ok: false, error: text || `Bridge returned HTTP ${response.status}` };
  }

  if (!response.ok || !data.ok) {
    throw new Error(data.error || `Bridge returned HTTP ${response.status}`);
  }

  return data;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Failed to read local file."));
    reader.readAsDataURL(file);
  });
}

function fileBaseName(name) {
  return name.replace(/\.[^/.]+$/, "");
}

function isImageFile(file) {
  const supportedMimeTypes = new Set([
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp",
    "image/tiff"
  ]);

  if (file.type && supportedMimeTypes.has(file.type.toLowerCase())) {
    return true;
  }

  return /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(file.name || "");
}

function queueCounts() {
  const total = queue.length;
  const done = queue.filter((item) => item.status === "done").length;
  const error = queue.filter((item) => item.status === "error").length;
  const pending = queue.filter((item) => item.status === "pending").length;
  return { total, done, error, pending };
}

function renderQueueSummary() {
  const counts = queueCounts();
  refs.queueSummary.textContent = `Queue: ${counts.total} item(s) - Pending ${counts.pending}, Done ${counts.done}, Error ${counts.error}`;
}

function renderResults() {
  refs.resultLinks.innerHTML = "";
  const doneItems = queue.filter((item) => item.status === "done" && item.resultDataUrl);
  if (!doneItems.length) {
    const empty = document.createElement("p");
    empty.className = "emptyState";
    empty.textContent = "Processed downloads will appear here.";
    refs.resultLinks.appendChild(empty);
    return;
  }

  doneItems.forEach((item) => {
    const a = document.createElement("a");
    a.className = "resultLink";
    a.href = item.resultDataUrl;
    a.download = item.outputName;
    a.textContent = `Download ${item.outputName}`;
    refs.resultLinks.appendChild(a);
  });
}

function setBusyUIState(busy) {
  refs.runUpscale.disabled = busy;
  refs.clearQueue.disabled = busy;
  refs.imageInput.disabled = busy;
  refs.checkConnection.disabled = busy;
  refs.dropZone.setAttribute("aria-disabled", busy ? "true" : "false");
}

async function saveSettings() {
  const payload = {
    backendUrl: refs.backendUrl.value,
    scale: refs.scale.value
  };
  await chrome.storage.local.set({ [SETTINGS_KEY]: payload });
}

async function loadSettings() {
  const { [SETTINGS_KEY]: saved } = await chrome.storage.local.get([SETTINGS_KEY]);
  if (!saved) {
    return;
  }

  refs.backendUrl.value = saved.backendUrl || refs.backendUrl.value;
  refs.scale.value = saved.scale || refs.scale.value;
}

async function checkBackendHealth() {
  setStatus("Checking local bridge...", "info");
  try {
    const baseUrl = await resolveReachableBridge();
    await saveSettings();
    setStatus(`Bridge is online and ready at ${baseUrl}.`, "ok");
  } catch (error) {
    setStatus(`Bridge check failed: ${toBridgeError(error)}`, "error");
  }
}

function updateLatestDownload(resultDataUrl, outputName) {
  refs.downloadResult.href = resultDataUrl;
  refs.downloadResult.download = outputName;
  refs.downloadResult.classList.remove("hidden");
}

function addFilesToQueue(files) {
  const inputFiles = Array.from(files || []);
  const accepted = inputFiles.filter(isImageFile);
  const rejectedCount = inputFiles.length - accepted.length;

  if (!accepted.length) {
    setStatus("No supported image files found. Use PNG, JPG, WEBP, BMP, or TIFF.", "warn");
    return;
  }

  for (const file of accepted) {
    queue.push({
      id: nextJobId,
      file,
      status: "pending",
      resultDataUrl: null,
      outputName: "",
      error: ""
    });
    nextJobId += 1;
  }

  renderQueueSummary();
  renderResults();
  if (rejectedCount > 0) {
    setStatus(
      `Added ${accepted.length} image(s). Skipped ${rejectedCount} unsupported file(s).`,
      "warn"
    );
    return;
  }

  setStatus(`Added ${accepted.length} image(s) to queue.`, "ok");
}

async function processQueue() {
  if (isProcessing) {
    return;
  }

  const jobs = queue.filter((item) => item.status === "pending" || item.status === "error");
  if (!jobs.length) {
    setStatus("Queue has no pending jobs. Add images or clear and re-queue.", "warn");
    return;
  }

  let baseUrl;
  try {
    baseUrl = await resolveReachableBridge();
  } catch (error) {
    setStatus(`Bridge check failed: ${toBridgeError(error)}`, "error");
    return;
  }

  isProcessing = true;
  setBusyUIState(true);
  refs.downloadResult.classList.add("hidden");

  let failures = 0;
  let firstFailureReason = "";
  try {
    await saveSettings();

    for (let index = 0; index < jobs.length; index += 1) {
      const job = jobs[index];
      job.status = "processing";
      job.error = "";
      renderQueueSummary();

      setStatus(
        `Processing ${index + 1}/${jobs.length}: ${job.file.name}. First item may take longer due to JIT.`,
        "info"
      );

      try {
        const sourceDataUrl = await fileToDataUrl(job.file);

        const payload = {
          image_base64: sourceDataUrl,
          scale: parseIntegerInput(refs.scale, 2)
        };

        const result = await postUpscale(baseUrl, payload);

        job.status = "done";
        job.resultDataUrl = result.image_base64;
        job.outputName = `${fileBaseName(job.file.name)}_upscaled_x${payload.scale}.png`;

        refs.upscaledPreview.src = result.image_base64;
        refs.upscaledPreview.style.display = "block";
        updateLatestDownload(result.image_base64, job.outputName);
      } catch (error) {
        failures += 1;
        job.status = "error";
        job.error = toBridgeError(error);
        if (!firstFailureReason) {
          firstFailureReason = job.error;
        }
      }

      renderQueueSummary();
      renderResults();
    }

    const doneCount = queue.filter((item) => item.status === "done").length;
    if (failures > 0) {
      if (doneCount === 0) {
        setStatus(`Batch failed. ${firstFailureReason}`, "error");
        return;
      }

      setStatus(
        `Batch finished with ${failures} failure(s). First error: ${firstFailureReason}`,
        "warn"
      );
    } else {
      setStatus(`Batch complete. ${doneCount} image(s) upscaled successfully.`, "ok");
    }
  } finally {
    isProcessing = false;
    setBusyUIState(false);
  }
}

function clearQueue() {
  if (isProcessing) {
    return;
  }

  queue.length = 0;
  refs.resultLinks.innerHTML = "";
  refs.upscaledPreview.removeAttribute("src");
  refs.upscaledPreview.style.display = "none";
  refs.downloadResult.classList.add("hidden");

  renderQueueSummary();
  renderResults();
  setStatus("Queue cleared.", "info");
}

function setupDropZone() {
  refs.dropZone.addEventListener("click", () => {
    if (!isProcessing) {
      refs.imageInput.click();
    }
  });

  refs.dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (!isProcessing) {
        refs.imageInput.click();
      }
    }
  });

  const dragEvents = ["dragenter", "dragover"];
  for (const eventName of dragEvents) {
    refs.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      refs.dropZone.classList.add("dragover");
    });
  }

  const leaveEvents = ["dragleave", "drop"];
  for (const eventName of leaveEvents) {
    refs.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      refs.dropZone.classList.remove("dragover");
    });
  }

  refs.dropZone.addEventListener("drop", (event) => {
    if (isProcessing) {
      return;
    }
    addFilesToQueue(event.dataTransfer?.files || []);
  });
}

refs.checkConnection.addEventListener("click", checkBackendHealth);
refs.runUpscale.addEventListener("click", processQueue);
refs.clearQueue.addEventListener("click", clearQueue);

refs.imageInput.addEventListener("change", () => {
  if (isProcessing) {
    return;
  }
  addFilesToQueue(refs.imageInput.files || []);
  refs.imageInput.value = "";
});

setupDropZone();
renderQueueSummary();
renderResults();

loadSettings()
  .then(() => {
    setStatus("Ready. Add images, check bridge, and process queue.", "info");
  })
  .catch((error) => {
    setStatus(`Could not load local settings: ${error.message}`, "warn");
  });
