# Talk demo

Offline playback of a dynamic Gaussian splat scene in Chrome. No install beyond Python 3.

From this directory:

```bash
python3 serve.py
```

Then open [http://127.0.0.1:5173](http://127.0.0.1:5173).

`serve.py` answers HTTP Range requests with status 206. The system Python on this Mac is 3.9, and `python3 -m http.server` ignores Range, so the viewer would download each file in one response instead of fetching the header, the static section, and each group of pictures on their own.

The root URL loads `flame.splat4d` (333,824 splats, 40 frames, 18.5 MB). Drag to orbit, scroll to zoom, and use the timeline to scrub. The readout at the top left is bytes loaded, decode time, sort time, and frame rate.

The smaller encode of the same scene is a second bookmark:

[http://127.0.0.1:5173/?file=hero.splat4d](http://127.0.0.1:5173/?file=hero.splat4d)

If the page does not render, add `&gl=1` to force the WebGL2 path.

`python3 -m http.server` answers HTTP Range requests, which is how this file is read. Do not open `index.html` as a file URL.
