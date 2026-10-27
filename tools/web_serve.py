"""Local server for the website (tunnel-friendly): threaded, HTTP/1.1 keep-alive, a deep accept queue, JavaScript
modules and WASM with the right types, no caching (edits show on reload).

    python3 tools/web_serve.py site 8001          # http://localhost:8001 (127.0.0.1 only)

``python -m http.server`` closes every connection and queues 5 pending ones, so the page's burst of parallel
downloads through an SSH tunnel can drop requests ("Failed to fetch dynamically imported module").
"""

from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript", ".mjs": "text/javascript",
                      ".wasm": "application/wasm", ".json": "application/json", ".svg": "image/svg+xml", ".woff2": "font/woff2"}  # fmt: skip

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # errors only
        if args and str(args[1])[:1] in "45":
            super().log_message(fmt, *args)


class Server(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root")
    parser.add_argument("port", type=int, nargs="?", default=8000)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    with Server((args.bind, args.port), functools.partial(Handler, directory=args.root)) as httpd:
        print(f"serving {args.root} on http://{args.bind}:{args.port}", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
