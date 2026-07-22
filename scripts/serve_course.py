"""Serve the course locally with the same paths used by GitHub Pages."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SITE_ROOT = REPOSITORY_ROOT / "site"


class CourseRequestHandler(SimpleHTTPRequestHandler):
    """Serve site assets at / and repository sources at /src and /tests."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(SITE_ROOT), **kwargs)

    def translate_path(self, path: str) -> str:
        request_path = urlsplit(path).path
        directory = self.directory
        if request_path in {"/src", "/tests"} or request_path.startswith(
            ("/src/", "/tests/")
        ):
            self.directory = str(REPOSITORY_ROOT)
        try:
            return super().translate_path(path)
        finally:
            self.directory = directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), CourseRequestHandler)
    print(f"Serving course at http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
