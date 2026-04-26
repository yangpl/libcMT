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
from discretize.mixins.mpl_mod import Slicer
from matplotlib.colors import SymLogNorm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the latest inversion gradient from gradient.h5."
    )
    parser.add_argument(
        "gradient_file",
        nargs="?",
        default="gradient.h5",
        help="Path to the gradient HDF5 file. Default: gradient.h5",
    )
    parser.add_argument(
        "--mesh-model",
        default="model_init.h5",
        help="Reference model file that provides fx1/fx2/fx3 when the gradient file omits them. Default: model_init.h5",
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


def read_mesh(path):
    with h5py.File(path, "r") as handle:
        x1 = np.asarray(handle["fx1"][:], dtype=np.float64)
        x2 = np.asarray(handle["fx2"][:], dtype=np.float64)
        x3 = np.asarray(handle["fx3"][:], dtype=np.float64)
    return x1, x2, x3


def read_gradient(path, component, mesh_path):
    with h5py.File(path, "r") as handle:
        if all(name in handle for name in ("fx1", "fx2", "fx3")):
            x1 = np.asarray(handle["fx1"][:], dtype=np.float64)
            x2 = np.asarray(handle["fx2"][:], dtype=np.float64)
            x3 = np.asarray(handle["fx3"][:], dtype=np.float64)
        else:
            x1, x2, x3 = read_mesh(mesh_path)
        gradient = np.asarray(handle[component][:], dtype=np.float64)
        iteration = int(handle["iteration"][0]) if "iteration" in handle else None
    return x1, x2, x3, gradient, iteration


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
    x1, x2, x3, gradient, iteration = read_gradient(gradient_path, args.component, args.mesh_model)
    mesh, x3_plot = build_plotting_mesh(x1, x2, x3)
    values = reorder_gradient_for_plotting(gradient)

    vmax = args.vmax if args.vmax is not None else robust_symmetric_limit(values)
    linthresh = args.linthresh if args.linthresh is not None else max(vmax * 1e-3, 1e-30)

    print(mesh)
    print(
        f"Loaded {args.component} from {gradient_path}: "
        f"nx={x1.size - 1}, ny={x2.size - 1}, nz={x3.size - 1}, "
        f"vmax={vmax:g}, linthresh={linthresh:g}"
        + (f", iteration={iteration}" if iteration is not None else "")
    )

    fig = plt.figure()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        tracker = Slicer(
            mesh,
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
    fig.canvas.mpl_connect("scroll_event", tracker.onscroll)
    title = f"{args.component}: {gradient_path.name}"
    if iteration is not None:
        title += f" (iteration {iteration})"
    fig.suptitle(title)

    output = Path(args.save) if args.save else gradient_path.with_name(
        f"{gradient_path.stem}_{args.component}.png"
    )
    fig.canvas.draw()
    fig.savefig(output, dpi=180)
    print(f"Saved gradient figure to {output}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
