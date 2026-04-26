#!/usr/bin/env python3
"""Generate MT inversion true and initial models on the 3D modelling mesh.

The mesh and layered anisotropic background are matched to
``run_modelling/make_model_3d.py`` so inversion experiments use the same
domain size and baseline structure as the forward-modelling template.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np

NX_HALF = 50
NY_HALF = 50
NZ = 100
LX_HALF = 10000.0
LY_HALF = 10000.0
DX_MIN = 100.0
DY_MIN = 100.0
DZ = 50.0
OZ = 0.0
DEPTH = np.array([1000.0, 2300.0, 2400.0], dtype=np.float32)
RESISTIVITY = np.array([0.3, 1.0, 50.0, 2.0], dtype=np.float32)
ANISOTROPY = np.array([1.0, 1.0, 1.414, 1.0], dtype=np.float32)
BLOCK_RHO = 10.0
BLOCK_X = (-1500.0, 1500.0)
BLOCK_Y = (-1500.0, 1500.0)
BLOCK_Z = (1800.0, 2200.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create true and initial HDF5 resistivity models for the inversion template."
    )
    parser.add_argument(
        "--true-model",
        default="model_true.h5",
        help="Output path for the synthetic truth model. Default: model_true.h5",
    )
    parser.add_argument(
        "--initial-model",
        default="model_init.h5",
        help="Output path for the starting inversion model. Default: model_init.h5",
    )
    return parser.parse_args()


def create_nugrid(n: int, length: float, dx: float) -> np.ndarray:
    eps = 1e-15
    x = np.zeros(n + 1, dtype=np.float64)
    if abs(n * dx - length) < eps:
        x[:] = dx * np.arange(n + 1, dtype=np.float64)
        return x.astype(np.float32)

    r = 1.1
    while True:
        rr = (length * (r - 1.0) / dx + 1.0) ** (1.0 / n)
        if abs(rr - r) < eps:
            break
        r = rr

    x[0] = 0.0
    for i in range(1, n + 1):
        x[i] = (math.pow(r, i) - 1.0) * dx / (r - 1.0)
    return x.astype(np.float32)


def build_symmetric_nodes(nhalf: int, length_half: float, dmin: float) -> np.ndarray:
    xpos = create_nugrid(nhalf, length_half, dmin)
    return np.concatenate((-xpos[::-1], xpos[1:])).astype(np.float32)


def build_vertical_nodes(n: int, d: float, o: float) -> np.ndarray:
    return (o + d * np.arange(n + 1, dtype=np.float32)).astype(np.float32)


def find_index(x: np.ndarray, value: float) -> int:
    if value < x[0]:
        return 0
    i = np.searchsorted(x, value, side="right")
    k = max(0, i - 1)
    if k >= len(x) - 1:
        return len(x) - 2
    return k


def build_background_model(
    fx1: np.ndarray,
    fx2: np.ndarray,
    fx3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx = fx1.size - 1
    ny = fx2.size - 1
    nz = fx3.size - 1
    x3c = 0.5 * (fx3[:-1] + fx3[1:])

    rho11 = np.full((nz, ny, nx), RESISTIVITY[0], dtype=np.float32)
    rho22 = np.full((nz, ny, nx), RESISTIVITY[0], dtype=np.float32)
    rho33 = np.full((nz, ny, nx), RESISTIVITY[0] * (ANISOTROPY[0] ** 2), dtype=np.float32)

    for ilay, depth in enumerate(DEPTH):
        i3min = find_index(x3c, depth)
        rho11[i3min + 1 :, :, :] = RESISTIVITY[ilay + 1]
        rho22[i3min + 1 :, :, :] = RESISTIVITY[ilay + 1]
        rho33[i3min + 1 :, :, :] = RESISTIVITY[ilay + 1] * (ANISOTROPY[ilay + 1] ** 2)

    return rho11, rho22, rho33


def build_true_model(
    fx1: np.ndarray,
    fx2: np.ndarray,
    fx3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1c = 0.5 * (fx1[:-1] + fx1[1:])
    x2c = 0.5 * (fx2[:-1] + fx2[1:])
    x3c = 0.5 * (fx3[:-1] + fx3[1:])

    rho11, rho22, rho33 = build_background_model(fx1, fx2, fx3)
    mask_x = (x1c >= BLOCK_X[0]) & (x1c <= BLOCK_X[1])
    mask_y = (x2c >= BLOCK_Y[0]) & (x2c <= BLOCK_Y[1])
    mask_z = (x3c >= BLOCK_Z[0]) & (x3c <= BLOCK_Z[1])
    block_index = np.ix_(mask_z, mask_y, mask_x)
    rho11[block_index] = BLOCK_RHO
    rho22[block_index] = BLOCK_RHO
    rho33[block_index] = BLOCK_RHO
    return rho11, rho22, rho33


def write_model(
    path: Path,
    fx1: np.ndarray,
    fx2: np.ndarray,
    fx3: np.ndarray,
    rho11: np.ndarray,
    rho22: np.ndarray,
    rho33: np.ndarray,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fx1", data=fx1)
        handle.create_dataset("fx2", data=fx2)
        handle.create_dataset("fx3", data=fx3)
        handle.create_dataset("frho11", data=rho11)
        handle.create_dataset("frho22", data=rho22)
        handle.create_dataset("frho33", data=rho33)


def main() -> None:
    args = parse_args()
    fx1 = build_symmetric_nodes(NX_HALF, LX_HALF, DX_MIN)
    fx2 = build_symmetric_nodes(NY_HALF, LY_HALF, DY_MIN)
    fx3 = build_vertical_nodes(NZ, DZ, OZ)

    rho11_true, rho22_true, rho33_true = build_true_model(fx1, fx2, fx3)
    rho11_init, rho22_init, rho33_init = build_background_model(fx1, fx2, fx3)

    write_model(Path(args.true_model), fx1, fx2, fx3, rho11_true, rho22_true, rho33_true)
    write_model(Path(args.initial_model), fx1, fx2, fx3, rho11_init, rho22_init, rho33_init)

    print(
        f"Wrote {args.true_model} and {args.initial_model} "
        f"(nx={fx1.size - 1}, ny={fx2.size - 1}, nz={fx3.size - 1})."
    )


if __name__ == "__main__":
    main()
