#!/usr/bin/env python3
import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np
from discretize import TensorMesh
from matplotlib.colors import SymLogNorm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize an inversion gradient snapshot from gradient_iterXXXX.h5."
    )
    parser.add_argument(
        "gradient_file",
        nargs="?",
        default="gradient_iter0000.h5",
        help="Path to the gradient HDF5 file. Default: gradient_iter0000.h5",
    )
    parser.add_argument(
        "--component",
        choices=("grad_mh", "grad_mv"),
        default="grad_mh",
        help="Gradient component to plot. Default: grad_mh",
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
        "--vmax",
        type=float,
        default=None,
        help="Symmetric color limit. Default: 98th percentile of |gradient|",
    )
    parser.add_argument(
        "--linthresh",
        type=float,
        default=None,
        help="Linear threshold for SymLogNorm. Default: vmax * 1e-3",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Output PNG filename. Default: <gradient_file>_<component>.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving. Default: save only",
    )
    return parser.parse_args()


def read_gradient(path, component):
    with h5py.File(path, "r") as handle:
        x1 = np.asarray(handle["fx1"][:], dtype=np.float64)
        x2 = np.asarray(handle["fx2"][:], dtype=np.float64)
        x3 = np.asarray(handle["fx3"][:], dtype=np.float64)
        gradient = np.asarray(handle[component][:], dtype=np.float64)
    return x1, x2, x3, gradient


def build_plotting_mesh(x1, x2, x3):
    x3_plot = -x3[::-1]
    mesh = TensorMesh(
        [np.diff(x1), np.diff(x2), np.diff(x3_plot)],
        origin=(x1[0], x2[0], x3_plot[0]),
    )
    return mesh, x3_plot


def reorder_gradient_for_plotting(gradient):
    # HDF5 gradients use (nz, ny, nx). TensorMesh slicer expects x-fastest layout.
    return np.transpose(gradient, (2, 1, 0))[:, :, ::-1]


def robust_symmetric_limit(values):
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    vmax = np.percentile(finite, 98.0)
    return float(vmax) if vmax > 0.0 else 1.0


def main():
    args = parse_args()
    gradient_path = Path(args.gradient_file)
    x1, x2, x3, gradient = read_gradient(gradient_path, args.component)
    mesh, x3_plot = build_plotting_mesh(x1, x2, x3)
    values = reorder_gradient_for_plotting(gradient)

    vmax = args.vmax if args.vmax is not None else robust_symmetric_limit(values)
    linthresh = args.linthresh if args.linthresh is not None else max(vmax * 1e-3, 1e-30)

    print(mesh)
    print(
        f"Loaded {args.component} from {gradient_path}: "
        f"nx={x1.size - 1}, ny={x2.size - 1}, nz={x3.size - 1}, "
        f"vmax={vmax:g}, linthresh={linthresh:g}"
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive.*",
            category=UserWarning,
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
                "norm": SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax),
                "cmap": "seismic",
            },
        )
    fig = plt.gcf()
    fig.suptitle(f"{args.component}: {gradient_path.name}")

    output = Path(args.save) if args.save else gradient_path.with_name(
        f"{gradient_path.stem}_{args.component}.png"
    )
    fig.savefig(output, dpi=180)
    print(f"Saved gradient figure to {output}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
