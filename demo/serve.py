#!/usr/bin/env python3
"""Static file server that honors single HTTP Range requests.

The system Python on this Mac (3.9) ignores Range and answers 200 with the
whole file. The viewer decides it is in range mode only when the status is
206, then fetches the header, the static section, and each group of pictures
separately. Run this instead of `python3 -m http.server`.
"""

import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        header = self.headers.get("Range")
        if not header or not header.startswith("bytes=") or "," in header:
            return super().do_GET()
        path = self.translate_path(self.path.split("?", 1)[0])
        if not os.path.isfile(path):
            return super().do_GET()
        size = os.path.getsize(path)
        spec = header[6:].strip()
        if "-" not in spec:
            self.send_error(400, "Bad Range")
            return
        start_s, end_s = spec.split("-", 1)
        try:
            if start_s == "":
                suffix = int(end_s)
                if suffix <= 0:
                    raise ValueError
                start = max(0, size - suffix)
                end = size - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else size - 1
        except ValueError:
            self.send_error(400, "Bad Range")
            return
        if start >= size or start < 0 or end < start:
            self.send_error(416, "Range Not Satisfiable")
            return
        end = min(end, size - 1)
        length = end - start + 1
        try:
            fh = open(path, "rb")
        except OSError:
            self.send_error(404, "Not Found")
            return
        try:
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            fh.seek(start)
            remaining = length
            while remaining:
                chunk = fh.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        finally:
            fh.close()


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    server = ThreadingHTTPServer(("127.0.0.1", 5173), RangeHandler)
    print(f"Serving {root} at http://127.0.0.1:5173", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
