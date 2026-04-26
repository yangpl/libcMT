#!/usr/bin/env python3
"""Rebuild an HDF5 resistivity model from the optimizer's final-x log output."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create model_recovered.h5 from inversion.log.")
    parser.add_argument("reference_model", nargs="?", default="model_init.h5")
    parser.add_argument("inversion_log", nargs="?", default="inversion.log")
    parser.add_argument("output_model", nargs="?", default="model_recovered.h5")
    return parser.parse_args()


def read_final_x(path: Path) -> np.ndarray:
    text = path.read_text()
    match = re.search(r"final x:\s*(.*)", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find 'final x:' in {path}")
    data = np.fromstring(match.group(1), sep=" ", dtype=np.float32)
    if data.size == 0:
        raise ValueError(f"No optimizer values parsed from {path}")
    return data


def main() -> None:
    args = parse_args()
    final_x = read_final_x(Path(args.inversion_log))

    with h5py.File(args.reference_model, "r") as src:
        fx1 = src["fx1"][:]
        fx2 = src["fx2"][:]
        fx3 = src["fx3"][:]
        nx = fx1.size - 1
        ny = fx2.size - 1
        nz = fx3.size - 1
        ncell = nx * ny * nz
        expected = 2 * ncell
        if final_x.size != expected:
            raise ValueError(f"Expected {expected} optimizer values, found {final_x.size}")

        rho11 = np.exp(-final_x[:ncell]).reshape((nz, ny, nx)).astype(np.float32)
        rho33 = np.exp(-final_x[ncell:]).reshape((nz, ny, nx)).astype(np.float32)
        rho22 = rho11.copy()

    with h5py.File(args.output_model, "w") as dst:
        dst.create_dataset("fx1", data=fx1)
        dst.create_dataset("fx2", data=fx2)
        dst.create_dataset("fx3", data=fx3)
        dst.create_dataset("frho11", data=rho11)
        dst.create_dataset("frho22", data=rho22)
        dst.create_dataset("frho33", data=rho33)

    print(f"Wrote recovered model to {args.output_model}")


if __name__ == "__main__":
    main()
