#!/usr/bin/env python3
from __future__ import annotations

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

NX = 3
NY = 3
X1MIN = -9000.0
X1MAX = 9000.0
X2MIN = -9000.0
X2MAX = 9000.0
X3 = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small MT receiver grid and plot the station layout."
    )
    parser.add_argument(
        "--receiver-file",
        default="receivers.h5",
        help="Output HDF5 receiver file. Default: receivers.h5",
    )
    parser.add_argument(
        "--plot-file",
        default="receivers_layout.png",
        help="Output receiver layout plot. Default: receivers_layout.png",
    )
    return parser.parse_args()


def build_receiver_grid() -> np.ndarray:
    x_values = np.linspace(X1MIN, X1MAX, NX, dtype=np.float32)
    y_values = np.linspace(X2MIN, X2MAX, NY, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    return np.column_stack(
        (
            x_grid.ravel(),
            y_grid.ravel(),
            np.full(x_grid.size, X3, dtype=np.float32),
        )
    ).astype(np.float32)


def write_receivers(path: str, receiver_position: np.ndarray) -> None:
    nrec = receiver_position.shape[0]
    receiver_azimuth = np.zeros(nrec, dtype=np.float32)
    receiver_dip = np.zeros(nrec, dtype=np.float32)
    receiver_index = np.arange(1, nrec + 1, dtype=np.int32)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("receiver_position", data=receiver_position)
        handle.create_dataset("receiver_azimuth", data=receiver_azimuth)
        handle.create_dataset("receiver_dip", data=receiver_dip)
        handle.create_dataset("receiver_index", data=receiver_index)


def plot_receivers(path: str, receiver_position: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    ax.scatter(
        receiver_position[:, 0],
        receiver_position[:, 1],
        s=65,
        c=receiver_position[:, 2],
        cmap="viridis",
        edgecolors="black",
        linewidths=0.6,
    )
    for idx, (x_pos, y_pos, _) in enumerate(receiver_position, start=1):
        ax.text(x_pos, y_pos + 90.0, str(idx), ha="center", va="bottom", fontsize=8)

    ax.set_title("Receiver Layout")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(X1MIN - 1000.0, X1MAX + 1000.0)
    ax.set_ylim(X2MIN - 1000.0, X2MAX + 1000.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    receiver_position = build_receiver_grid()
    write_receivers(args.receiver_file, receiver_position)
    plot_receivers(args.plot_file, receiver_position)
    print(
        f"Wrote {receiver_position.shape[0]} receivers to {args.receiver_file} "
        f"and saved layout plot to {args.plot_file}."
    )


if __name__ == "__main__":
    main()
