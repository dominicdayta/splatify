#!/usr/bin/env python3
"""Convert a 3D Gaussian Splatting PLY into an antimatter15 .splat file.

The record layout, the sort, and the spherical-harmonic degree-0 color
conversion follow convert.py from https://github.com/antimatter15/splat
Copyright (c) 2023 Kevin Kwok, MIT License.
Each splat is 32 bytes: position (3 float32), scale (3 float32),
RGBA (4 uint8), quaternion (4 uint8).
"""

from __future__ import annotations

import argparse
from pathlib import Path

# SH degree-0 basis constant used by the original 3DGS paper and by antimatter15.
SH_C0 = 0.28209479177387814


def _libs():
    import numpy as np
    from plyfile import PlyData

    return np, PlyData


def splat_dtype():
    np, _ = _libs()
    return np.dtype(
        [
            ("pos", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("rgba", "u1", (4,)),
            ("rot", "u1", (4,)),
        ]
    )


def process_ply_to_splat(ply_file_path: str | Path) -> bytes:
    """Return the .splat bytes for one 3DGS PLY."""
    np, PlyData = _libs()
    record_type = splat_dtype()
    if record_type.itemsize != 32:
        raise RuntimeError("splat record must be 32 bytes")

    plydata = PlyData.read(str(ply_file_path))
    vert = plydata["vertex"].data
    order = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1.0 + np.exp(-vert["opacity"].astype(np.float64)))
    )
    vert = vert[order]

    position = np.stack([vert["x"], vert["y"], vert["z"]], axis=1).astype(np.float32)
    scales = np.exp(
        np.stack(
            [vert["scale_0"], vert["scale_1"], vert["scale_2"]], axis=1
        ).astype(np.float32)
    )
    rotation = np.stack(
        [vert["rot_0"], vert["rot_1"], vert["rot_2"], vert["rot_3"]], axis=1
    ).astype(np.float32)
    norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    norm = np.maximum(norm, np.float32(1e-12))
    rotation = rotation / norm

    color = np.stack(
        [
            0.5 + SH_C0 * vert["f_dc_0"],
            0.5 + SH_C0 * vert["f_dc_1"],
            0.5 + SH_C0 * vert["f_dc_2"],
            1.0 / (1.0 + np.exp(-vert["opacity"].astype(np.float64))),
        ],
        axis=1,
    )
    rgba = (color * 255.0).clip(0, 255).astype(np.uint8)
    quaternion = (rotation * 128.0 + 128.0).clip(0, 255).astype(np.uint8)

    records = np.empty(len(vert), dtype=record_type)
    records["pos"] = position
    records["scale"] = scales
    records["rgba"] = rgba
    records["rot"] = quaternion
    return records.tobytes()


def save_splat_file(splat_data: bytes, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(splat_data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a 3DGS PLY file to an antimatter15 .splat file."
    )
    parser.add_argument("ply", help="Input PLY written by 3DGS or gsplat.")
    parser.add_argument("-o", "--output", required=True, help="Output .splat path.")
    args = parser.parse_args()
    splat = process_ply_to_splat(args.ply)
    save_splat_file(splat, args.output)
    count = len(splat) // splat_dtype().itemsize
    print(f"Wrote {args.output} ({count} splats)")


if __name__ == "__main__":
    main()
