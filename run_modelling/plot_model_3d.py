#!/usr/bin/env python3
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from discretize import TensorMesh
from matplotlib.colors import LogNorm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a resistivity model from model.h5 with a 3D slicer."
    )
    parser.add_argument(
        "model_file",
        nargs="?",
        default="model.h5",
        help="Path to the model HDF5 file. Default: model.h5",
    )
    parser.add_argument(
        "--component",
        choices=("frho11", "frho22", "frho33"),
        default="frho33",
        help="Resistivity tensor component to plot. Default: frho33",
    )
    parser.add_argument(
        "--xslice",
        type=float,
        default=0.0,
        help="Slice position in x. Default: 0",
    )
    parser.add_argument(
        "--yslice",
        type=float,
        default=0.0,
        help="Slice position in y. Default: 0",
    )
    parser.add_argument(
        "--zslice",
        type=float,
        default=-2050.0,
        help="Slice position in z for the plotting mesh. Default: -2050",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.1,
        help="Lower color limit for LogNorm. Default: 0.1",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=100.0,
        help="Upper color limit for LogNorm. Default: 100",
    )
    return parser.parse_args()


def read_model(path, component):
    with h5py.File(path, "r") as handle:
        x1 = np.asarray(handle["fx1"][:], dtype=np.float64)
        x2 = np.asarray(handle["fx2"][:], dtype=np.float64)
        x3 = np.asarray(handle["fx3"][:], dtype=np.float64)
        rho = np.asarray(handle[component][:], dtype=np.float64)
    return x1, x2, x3, rho


def build_plotting_mesh(x1, x2, x3):
    x3_plot = -x3[::-1]
    hx = np.diff(x1)
    hy = np.diff(x2)
    hz = np.diff(x3_plot)
    mesh = TensorMesh([hx, hy, hz], origin=(x1[0], x2[0], x3_plot[0]))
    return mesh, x3_plot


def reorder_model_for_plotting(rho):
    # HDF5 model uses (nz, ny, nx). TensorMesh slicer expects x-fastest layout.
    values = np.transpose(rho, (2, 1, 0))
    return values[:, :, ::-1]


def main():
    args = parse_args()
    x1, x2, x3, rho = read_model(args.model_file, args.component)
    mesh, x3_plot = build_plotting_mesh(x1, x2, x3)
    values = reorder_model_for_plotting(rho)

    print(mesh)
    print(
        f"Loaded {args.component} from {args.model_file}: "
        f"nx={x1.size - 1}, ny={x2.size - 1}, nz={x3.size - 1}"
    )

    mesh.plot_3d_slicer(
        values,
        xslice=args.xslice,
        yslice=args.yslice,
        zslice=args.zslice,
        xlim=(x1[0], x1[-1]),
        ylim=(x2[0], x2[-1]),
        zlim=(x3_plot[0], x3_plot[-1]),
        pcolor_opts={
            "norm": LogNorm(vmin=args.vmin, vmax=args.vmax),
            "cmap": "jet",
        },
    )
    plt.show()


if __name__ == "__main__":
    main()
