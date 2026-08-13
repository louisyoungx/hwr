"""Serve the read-only local dashboard for one foundation training run."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

from hwr.train.foundation_dashboard import (
    DASHBOARD_HTML,
    load_dashboard_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def handler_for(run_path: Path) -> type[BaseHTTPRequestHandler]:
    target = run_path.resolve()

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                self._respond(DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
                return
            if self.path == "/api/snapshot":
                payload = json.dumps(
                    load_dashboard_snapshot(target), ensure_ascii=False
                ).encode()
                self._respond(payload, "application/json; charset=utf-8")
                return
            self.send_error(404)

        def _respond(self, payload: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, message: str, *args: object) -> None:
            del message, args

    return DashboardHandler


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if not arguments.run_path.is_dir():
        raise FileNotFoundError(arguments.run_path)
    if not 1 <= arguments.port <= 65535:
        raise ValueError("dashboard port is invalid")
    server = ThreadingHTTPServer(
        (arguments.host, arguments.port), handler_for(arguments.run_path)
    )
    print(f"http://{arguments.host}:{arguments.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
