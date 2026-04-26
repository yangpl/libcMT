#!/usr/bin/env python3
import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from discretize import TensorMesh
from matplotlib.colors import LogNorm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize true and initial inversion models from HDF5 as 3D voxel views."
    )
    parser.add_argument(
        "--true-model",
        default="model_true.h5",
        help="Path to the true-model HDF5 file. Default: model_true.h5",
    )
    parser.add_argument(
        "--initial-model",
        default="model_init.h5",
        help="Path to the initial-model HDF5 file. Default: model_init.h5",
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
        default=-2400.0,
        help="Cutaway position in z. Default: -2400",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Coarsening factor for 3D voxel plotting. Default: 4",
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
    parser.add_argument(
        "--save-prefix",
        default="model_3d",
        help="Filename prefix for saved PNG files. Default: model_3d",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures after saving. Default: save only",
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
    # HDF5 model uses (nz, ny, nx). The plotting arrays use (nx, ny, nz).
    values = np.transpose(rho, (2, 1, 0))
    return values[:, :, ::-1]


def coarsen_edges(edges, stride):
    indices = list(range(0, edges.size - 1, stride))
    if indices[-1] != edges.size - 1:
        indices.append(edges.size - 1)
    return edges[indices], indices


def coarsen_values(values, x_indices, y_indices, z_indices):
    coarse = np.empty(
        (len(x_indices) - 1, len(y_indices) - 1, len(z_indices) - 1),
        dtype=np.float64,
    )
    for i in range(coarse.shape[0]):
        for j in range(coarse.shape[1]):
            for k in range(coarse.shape[2]):
                block = values[
                    x_indices[i] : x_indices[i + 1],
                    y_indices[j] : y_indices[j + 1],
                    z_indices[k] : z_indices[k + 1],
                ]
                positive = block[block > 0.0]
                coarse[i, j, k] = np.exp(np.mean(np.log(positive)))
    return coarse


def voxel_alpha(values, norm):
    normalized = np.asarray(norm(values.ravel())).reshape(values.shape)
    normalized = np.clip(normalized, 0.0, 1.0)
    alpha = 0.12 + 0.58 * (1.0 - normalized)
    alpha[values <= 10.0] = np.maximum(alpha[values <= 10.0], 0.78)
    return alpha


def set_model_aspect(ax, x_edges, y_edges, z_edges):
    x_span = x_edges[-1] - x_edges[0]
    y_span = y_edges[-1] - y_edges[0]
    z_span = z_edges[-1] - z_edges[0]
    ax.set_box_aspect((x_span, y_span, 1.8 * z_span))


def plot_tensor_mesh_slicer(
    label,
    model_path,
    component,
    mesh,
    x1,
    x2,
    x3_plot,
    values,
    xslice,
    yslice,
    zslice,
    vmin,
    vmax,
):
    fig = plt.figure(figsize=(8.0, 7.0))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive.*",
            category=UserWarning,
        )
        mesh.plot_3d_slicer(
            values,
            xslice=xslice,
            yslice=yslice,
            zslice=zslice,
            xlim=(x1[0], x1[-1]),
            ylim=(x2[0], x2[-1]),
            zlim=(x3_plot[0], x3_plot[-1]),
            pcolor_opts={
                "norm": LogNorm(vmin=vmin, vmax=vmax),
                "cmap": "jet",
            },
            fig=fig,
        )
    fig.suptitle(f"{label} TensorMesh slicer: {Path(model_path).name}")
    return fig


def plot_one_model(label, model_path, component, xslice, yslice, zslice, stride, vmin, vmax):
    x1, x2, x3, rho = read_model(model_path, component)
    mesh, x3_plot = build_plotting_mesh(x1, x2, x3)
    values = reorder_model_for_plotting(rho)

    print(mesh)
    print(
        f"Loaded {component} from {model_path}: "
        f"nx={x1.size - 1}, ny={x2.size - 1}, nz={x3.size - 1}"
    )

    stride = max(1, stride)
    x_edges, x_indices = coarsen_edges(x1, stride)
    y_edges, y_indices = coarsen_edges(x2, stride)
    z_edges, z_indices = coarsen_edges(x3_plot, stride)
    coarse = coarsen_values(values, x_indices, y_indices, z_indices)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    xc, yc, zc = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

    cutaway = (xc < xslice) & (yc < yslice) & (zc > zslice)
    filled = np.isfinite(coarse) & (coarse > 0.0) & ~cutaway

    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("jet")
    normalized = np.asarray(norm(coarse.ravel())).reshape(coarse.shape)
    facecolors = cmap(normalized)
    facecolors[..., 3] = voxel_alpha(coarse, norm)

    xe, ye, ze = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")
    fig = plt.figure(figsize=(8.0, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(
        xe,
        ye,
        ze,
        filled,
        facecolors=facecolors,
        edgecolor=(0.05, 0.05, 0.05, 0.08),
        linewidth=0.15,
    )
    ax.set_title(f"{label}: {Path(model_path).name}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=24, azim=-135)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_zlim(z_edges[0], z_edges[-1])
    set_model_aspect(ax, x_edges, y_edges, z_edges)

    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    fig.colorbar(
        scalar_mappable,
        ax=ax,
        shrink=0.72,
        pad=0.02,
        label=f"{component} (Ohm m)",
    )
    slicer_fig = plot_tensor_mesh_slicer(
        label,
        model_path,
        component,
        mesh,
        x1,
        x2,
        x3_plot,
        values,
        xslice,
        yslice,
        zslice,
        vmin,
        vmax,
    )
    return fig, slicer_fig


def main():
    args = parse_args()
    figures = [
        (
            "True model",
            *plot_one_model(
                "True model",
                args.true_model,
                args.component,
                args.xslice,
                args.yslice,
                args.zslice,
                args.stride,
                args.vmin,
                args.vmax,
            ),
        ),
        (
            "Initial model",
            *plot_one_model(
                "Initial model",
                args.initial_model,
                args.component,
                args.xslice,
                args.yslice,
                args.zslice,
                args.stride,
                args.vmin,
                args.vmax,
            ),
        ),
    ]

    prefix = Path(args.save_prefix)
    for label, voxel_fig, slicer_fig in figures:
        suffix = "true" if label.startswith("True") else "initial"
        voxel_output = prefix.parent / f"{prefix.name}_{suffix}.png"
        slicer_output = prefix.parent / f"{prefix.name}_slicer_{suffix}.png"
        voxel_fig.savefig(voxel_output, dpi=180)
        slicer_fig.savefig(slicer_output, dpi=180)
        print(f"Saved {label.lower()} voxel figure to {voxel_output}")
        print(f"Saved {label.lower()} TensorMesh slicer figure to {slicer_output}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
