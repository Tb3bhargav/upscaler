# IBP + DCCI Chrome Sidebar Extension

This extension adds a Chrome side panel that calls your local Python upscaler.

## 1) Start the local Python bridge

From the project root:

```
.venv\Scripts\python.exe bridge_server.py --host 127.0.0.1 --port 8765
```

Health check endpoint:

```
http://127.0.0.1:8765/health
```

## 2) Load extension in Chrome Beta

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the `chrome_extension` folder in this project

## 3) Use the side panel

1. Click the extension icon to open the side panel
2. Keep bridge URL as `http://127.0.0.1:8765`
3. Click **Check bridge**
4. Add images using file picker or drag-and-drop zone
5. Click **Process Queue** to run batch upscale jobs
6. Download each output from **Batch Results** (or use **Download Latest PNG**)

## Notes

- Processing stays local on your machine.
- First run can be slower because Numba compiles kernels.
- IBP loop is fixed to 3 iterations in the backend.
- Failed jobs remain in queue with an error label so you can fix settings and retry.

## Troubleshooting: Bridge check failed

If the sidebar shows "Bridge check failed" or "Failed to fetch":

1. Start backend first:

```
.venv\Scripts\python.exe bridge_server.py --host 127.0.0.1 --port 8765
```

Keep this terminal open while using the extension.

2. Reload the unpacked extension in `chrome://extensions` only if you changed `manifest.json`.
3. In the sidebar, keep bridge URL as `http://127.0.0.1:8765` or `http://localhost:8765`.
4. Click **Check bridge** again.
