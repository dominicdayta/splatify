#!/usr/bin/env python3
"""Turn a short walk-around video into one .splat4d file.

Frames are capped, COLMAP estimates one pose per frame, gsplat trains a
static 3D Gaussian splat, and splat4d packs that single frame. A locked-off
shot of a moving subject will not reconstruct. Training is meant for the
Ubuntu machine with an RTX 3090 Ti.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK_ROOT = ROOT / "work"
OUT_ROOT = ROOT / "out"

# gsplat v1.5.3. The example trainer and the pip package need to match.
GSPLAT_COMMIT = "937e29912570c372bed6747a5c9bf85fed877bae"
GSPLAT_URL = "https://github.com/nerfstudio-project/gsplat.git"
GSPLAT_DIR = ROOT / "third_party" / "gsplat"


def run(cmd: list[str], *, cwd: Path | None = None, log: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    if log is None:
        try:
            subprocess.run(cmd, cwd=cwd, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}"
            ) from exc
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        code = process.wait()
    if code != 0:
        text = log.read_text(encoding="utf-8", errors="replace")
        if "out of memory" in text.lower():
            raise SystemExit(
                "Training ran out of GPU memory. "
                "Rerun the same command with --data-factor 4."
            )
        raise SystemExit(f"Command failed with exit code {code}. Log: {log}")


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"Required program {name} is not on PATH.")
    return path


def probe_video(video: Path) -> tuple[float, float]:
    """Return (duration_seconds, frames_per_second) for the first video stream."""
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(video),
        ],
        text=True,
    )
    payload = json.loads(raw)
    streams = payload.get("streams") or []
    if not streams:
        raise SystemExit(f"{video} has no video stream.")
    stream = streams[0]
    rate = stream.get("r_frame_rate") or "0/1"
    num, den = rate.split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    duration_text = (payload.get("format") or {}).get("duration")
    duration = float(duration_text) if duration_text not in (None, "N/A") else 0.0
    if duration <= 0 and fps > 0:
        nb = stream.get("nb_frames")
        if nb not in (None, "N/A"):
            duration = float(nb) / fps
    if duration <= 0 or fps <= 0:
        raise SystemExit(f"Could not read duration and frame rate from {video}.")
    return duration, fps


def extract_frames(video: Path, image_dir: Path, max_frames: int, max_edge: int) -> int:
    duration, fps = probe_video(video)
    target_fps = min(fps, max_frames / duration)
    # Commas inside the quotes belong to the scale expression, not the filter chain.
    scale = (
        "scale="
        f"'if(gte(iw,ih),min({max_edge},iw),-2)':"
        f"'if(gte(iw,ih),-2,min({max_edge},ih))'"
    )
    image_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"fps={target_fps},{scale}",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(image_dir / "frame_%04d.jpg"),
        ]
    )
    count = len(list(image_dir.glob("frame_*.jpg")))
    if count < 8:
        raise SystemExit(
            f"Extracted {count} frames. A walk-around needs more overlap than that. "
            "Use a longer clip."
        )
    print(f"Extracted {count} frames (cap {max_frames}, long edge {max_edge}).")
    return count


def link_factor_images(data_dir: Path, factor: int) -> None:
    """gsplat's factor>1 path expects images_<factor> to exist beside images/.

    When those files are JPEGs, the trainer downscales images/ itself.
    """
    if factor <= 1:
        return
    source = data_dir / "images"
    dest = data_dir / f"images_{factor}"
    dest.mkdir(parents=True, exist_ok=True)
    for image in sorted(source.glob("frame_*.jpg")):
        target = dest / image.name
        if target.exists():
            continue
        try:
            os.link(image, target)
        except OSError:
            shutil.copy2(image, target)


def run_colmap(data_dir: Path) -> None:
    database = data_dir / "database.db"
    images = data_dir / "images"
    sparse = data_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "SIMPLE_RADIAL",
        ]
    )
    run(
        [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(database),
            "--SequentialMatching.overlap",
            "10",
        ]
    )
    run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse),
        ]
    )
    model = choose_colmap_model(sparse)
    target = sparse / "0"
    if model != target:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(model, target)
    if _reconstruction_file(target, "images") is None:
        raise SystemExit(
            "COLMAP did not register a camera path. "
            "Use a slow walk around a still, textured object."
        )
    if _reconstruction_file(target, "points3D") is None:
        raise SystemExit("COLMAP wrote no 3D points. The match graph was too thin.")
    print(f"COLMAP model: {target}")


def _reconstruction_file(model: Path, stem: str) -> Path | None:
    for suffix in (".bin", ".txt"):
        candidate = model / f"{stem}{suffix}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def choose_colmap_model(sparse: Path) -> Path:
    models = [
        path
        for path in sparse.iterdir()
        if path.is_dir() and _reconstruction_file(path, "images") is not None
    ]
    if not models:
        raise SystemExit(
            "COLMAP did not produce a sparse model. "
            "Use a slow walk around a still, textured object."
        )

    def size_of(path: Path) -> int:
        found = _reconstruction_file(path, "images")
        return found.stat().st_size if found is not None else 0

    return max(models, key=size_of)


def ensure_gsplat() -> Path:
    trainer = GSPLAT_DIR / "examples" / "simple_trainer.py"
    if trainer.is_file():
        head = subprocess.check_output(
            ["git", "-C", str(GSPLAT_DIR), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if head != GSPLAT_COMMIT:
            raise SystemExit(
                f"{GSPLAT_DIR} is at {head}, and this pipeline expects {GSPLAT_COMMIT}. "
                "Remove splatify/third_party/gsplat and run again."
            )
    else:
        if GSPLAT_DIR.exists():
            raise SystemExit(
                f"{GSPLAT_DIR} exists and is not a gsplat checkout. Remove it and run again."
            )
        GSPLAT_DIR.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning gsplat {GSPLAT_COMMIT}")
        run(["git", "init", str(GSPLAT_DIR)])
        run(["git", "-C", str(GSPLAT_DIR), "remote", "add", "origin", GSPLAT_URL])
        run(
            [
                "git",
                "-C",
                str(GSPLAT_DIR),
                "fetch",
                "--depth",
                "1",
                "origin",
                GSPLAT_COMMIT,
            ]
        )
        run(["git", "-C", str(GSPLAT_DIR), "checkout", "FETCH_HEAD"])
    stamp = GSPLAT_DIR / ".splatify-example-deps"
    if not stamp.is_file():
        print("Installing the gsplat example requirements into this environment.")
        run(
            [sys.executable, "-m", "pip", "install", "-r", "examples/requirements.txt"],
            cwd=GSPLAT_DIR,
        )
        stamp.write_text("ok\n", encoding="utf-8")
    return trainer


def train(data_dir: Path, result_dir: Path, steps: int, data_factor: int) -> Path:
    trainer = ensure_gsplat()
    link_factor_images(data_dir, data_factor)
    result_dir.mkdir(parents=True, exist_ok=True)
    log = result_dir / "train.log"
    # eval_steps 0 never lands on a training step, so the run spends its time
    # fitting and still writes a ply on the last step because save_ply is set.
    run(
        [
            sys.executable,
            str(trainer),
            "default",
            "--data_dir",
            str(data_dir),
            "--data_factor",
            str(data_factor),
            "--result_dir",
            str(result_dir),
            "--max_steps",
            str(steps),
            "--save_ply",
            "--disable_viewer",
            "--disable_video",
            "--eval_steps",
            "0",
            "--test_every",
            "8",
        ],
        cwd=trainer.parent,
        log=log,
    )
    ply_dir = result_dir / "ply"
    plies = list(ply_dir.glob("point_cloud_*.ply")) if ply_dir.is_dir() else []
    if not plies:
        raise SystemExit(f"Training finished and wrote no ply under {ply_dir}.")
    return max(plies, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def pack(ply_path: Path, frames: Path, output: Path) -> None:
    from ply_to_splat import process_ply_to_splat, save_splat_file

    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    splat_path = frames / "frame_0000.splat"
    save_splat_file(process_ply_to_splat(ply_path), splat_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    run(["splat4d", "encode", "-i", str(frames), "-o", str(output)])


def scene_name(raw: str) -> str:
    name = raw.strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise SystemExit("-o takes a scene name, such as scene, and not a path.")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn a short walk-around video into one .splat4d file."
    )
    parser.add_argument("video", help="Input video of a still scene.")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Scene name. Writes splatify/out/<name>.splat4d.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=80,
        help="Most frames to keep (default 80). Fits COLMAP in 61 GB of RAM.",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=1600,
        help="Long-edge cap in pixels (default 1600).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=7000,
        help="gsplat steps. 7000 is a preview of about ten minutes. 30000 is a full fit.",
    )
    parser.add_argument(
        "--data-factor",
        type=int,
        default=2,
        help="gsplat image downsample. Use 4 if training runs out of GPU memory.",
    )
    args = parser.parse_args()

    if args.max_frames < 8:
        raise SystemExit("--max-frames must be at least 8.")
    if args.max_edge < 64:
        raise SystemExit("--max-edge must be at least 64.")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1.")
    if args.data_factor < 1:
        raise SystemExit("--data-factor must be at least 1.")

    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")
    name = scene_name(args.output)

    for tool in ("ffmpeg", "ffprobe", "colmap", "git", "splat4d"):
        require_tool(tool)

    data_dir = WORK_ROOT / name
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    extract_frames(video, data_dir / "images", args.max_frames, args.max_edge)
    run_colmap(data_dir)
    ply_path = train(
        data_dir,
        data_dir / "train",
        args.steps,
        args.data_factor,
    )
    output = OUT_ROOT / f"{name}.splat4d"
    pack(ply_path, data_dir / "splat_frames", output)
    print(f"Wrote {output}")
    print("Copy it next to the demo and open ?file=" + output.name)


if __name__ == "__main__":
    main()
