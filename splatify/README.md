# Video to splat4d

A short video of a still scene becomes one `.splat4d` file. Training runs on the Ubuntu machine (RTX 3090 Ti, 24 GB of VRAM, 61 GB of RAM). This Mac only needs the finished file. The talk viewer already knows how to open it.

Film a slow walk or orbit around something that holds still. COLMAP estimates one camera pose per frame from the background. A locked-off shot of a moving person will not produce a usable scene. True 4D reconstruction from one camera is out of scope for this pipeline.

```text
mp4 -> ffmpeg frames -> COLMAP poses -> gsplat -> ply -> frame_0000.splat -> scene.splat4d -> demo viewer
```

## Install

Do these in order, on the Ubuntu machine.

1. The NVIDIA driver. `nvidia-smi` should list the GeForce RTX 3090 Ti.
2. System packages:

```bash
sudo apt update
sudo apt install -y ffmpeg colmap git python3-venv python3-dev build-essential
```

3. A Python 3.10+ virtualenv, the CUDA PyTorch wheel, then this directory's requirements. PyTorch goes in first so a CPU wheel cannot sneak in. `requirements.txt` does not list torch.

```bash
cd splatify
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m pip install -r requirements.txt
splat4d --help
```

`splat4d --help` should print the encoder usage. If the CUDA 12.8 wheel does not match the driver `nvidia-smi` reports, install the matching wheel from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) before `requirements.txt`.

`gsplat` is pinned to 1.5.3, the same commit the script checks out for `examples/simple_trainer.py`. The example trainer also needs a COLMAP reader, OpenCV, and a fused SSIM build. The first real run installs those from the checkout. That step needs a network connection, a compiler, and the CUDA toolkit that matches the PyTorch wheel. It also downloads the LPIPS weights the example constructs at startup.

## Capture

Keep the clip inside what this machine can hold.

- A slow walk or orbit around a still, textured object, roughly 5 to 20 seconds.
- Overlap between moments, diffuse light, and a shutter fast enough to avoid blur.
- Stay inside the default caps. The script keeps at most 80 frames and scales the long edge to 1600 px. Eighty 1600 px images are the size that fits COLMAP in 61 GB of RAM. gsplat then trains at that resolution divided by `--data-factor` (2 by default, so about 800 px on the long edge), which fits in 24 GB. More frames, or 4K frames, are what exhaust the card.

## Run

From `splatify/`, with the virtualenv active:

```bash
python splatify.py clip.mp4 -o scene
```

That creates `work/scene/` and writes `out/scene.splat4d`. Running it again with the same name replaces `work/scene/`.

| Flag | Default | What it does |
| --- | --- | --- |
| `--max-frames` | 80 | Frame cap passed to ffmpeg. |
| `--max-edge` | 1600 | Long-edge cap in pixels. |
| `--steps` | 7000 | gsplat steps. About ten minutes on a 3090 Ti. `--steps 30000` is the full fit. |
| `--data-factor` | 2 | Extra downsample inside gsplat. |

On a CUDA out-of-memory error, rerun the same command with `--data-factor 4`.

The first run clones [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) at v1.5.3 (`937e29912570c372bed6747a5c9bf85fed877bae`) into `third_party/gsplat` and calls `examples/simple_trainer.py default` with `--save_ply`, `--data_factor`, and `--max_steps`. The viewer is left off so the run exits when the ply is written. `ply_to_splat.py` turns that ply into `frame_0000.splat` (position, `exp(scale)`, spherical-harmonic degree-0 color, quaternion), and `splat4d encode` writes one `.splat4d`. A single frame is a valid file: every splat is static, and the demo still orbits it.

`python splatify.py --help` is the check that does not need a GPU.

## View

Copy the file next to the other scenes and serve the demo from the repository root:

```bash
cp splatify/out/scene.splat4d demo/
cd demo && python3 serve.py
```

Open [http://127.0.0.1:5173/?file=scene.splat4d](http://127.0.0.1:5173/?file=scene.splat4d).

The viewer frames the scene from the file's bounds. Flame and hero keep their own bookmarks. `demo/serve.py` answers Range requests with status 206, which is how the file is read.
