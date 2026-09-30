from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from upscaler_main import upscale_ibp_dcci


def _decode_base64_image(data: str) -> bytes:
    payload = data.strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload, validate=True)


def _encode_png_data_url(image: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode output image as PNG.")
    b64 = base64.b64encode(png.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class UpscaleHandler(BaseHTTPRequestHandler):
    server_version = "IBPDCCIBridge/1.0"

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(200, {"ok": True, "service": "ibp-dcci-upscaler-bridge"})
            return
        self._send_json(404, {"ok": False, "error": "Route not found."})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/upscale":
            self._send_json(404, {"ok": False, "error": "Route not found."})
            return

        try:
            length_raw = self.headers.get("Content-Length", "0")
            content_length = int(length_raw)
            if content_length <= 0 or content_length > 50 * 1024 * 1024:
                raise ValueError("Invalid Content-Length (empty or >50MB).")
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))

            image_b64 = str(payload.get("image_base64", ""))
            if not image_b64:
                raise ValueError("image_base64 is required.")

            scale = int(payload.get("scale", 2))
            tensor_sigma = float(payload.get("tensor_sigma", 1.2))
            edge_threshold = float(payload.get("edge_threshold", 0.25))
            guided_radius = int(payload.get("guided_radius", 6))
            guided_eps = float(payload.get("guided_eps", 1e-3))

            if scale < 2:
                raise ValueError("scale must be >= 2")
            if guided_radius < 1:
                raise ValueError("guided_radius must be >= 1")

            image_bytes = _decode_base64_image(image_b64)
            image_arr = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_arr, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Unable to decode input image data.")

            output = upscale_ibp_dcci(
                image,
                scale=scale,
                tensor_sigma=tensor_sigma,
                edge_threshold=edge_threshold,
                guided_radius=guided_radius,
                guided_eps=guided_eps,
                ibp_iterations=3,
            )

            self._send_json(
                200,
                {
                    "ok": True,
                    "image_base64": _encode_png_data_url(output),
                    "width": int(output.shape[1]),
                    "height": int(output.shape[0]),
                },
            )
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Local HTTP bridge for IBP+DCCI upscaler")
    parser.add_argument("--host", default="127.0.0.1", type=str, help="Bind host")
    parser.add_argument("--port", default=8765, type=int, help="Bind port")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), UpscaleHandler)
    print(f"IBP+DCCI bridge listening on http://{args.host}:{args.port}")
    print("Open /health to verify the service is running.")
    server.serve_forever()


if __name__ == "__main__":
    main()
