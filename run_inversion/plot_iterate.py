#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot objective and gradient histories from iterate.txt.")
    parser.add_argument("iterate_file", nargs="?", default="iterate.txt")
    parser.add_argument("output_file", nargs="?", default="inversion_convergence.png")
    return parser.parse_args()


def read_iterates(path: Path) -> tuple[list[int], list[float], list[float]]:
    iterations: list[int] = []
    cost: list[float] = []
    grad_norm: list[float] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("=") or line.startswith("method:") or line.startswith("==>"):
            continue
        parts = line.split()
        if len(parts) != 7:
            continue
        try:
            iterations.append(int(parts[0]))
            cost.append(float(parts[1]))
            grad_norm.append(float(parts[3]))
        except ValueError:
            continue

    if not iterations:
        raise ValueError(f"No iteration rows found in {path}")
    return iterations, cost, grad_norm


def main() -> None:
    args = parse_args()
    iterations, cost, grad_norm = read_iterates(Path(args.iterate_file))

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True, constrained_layout=True)
    fig.suptitle("Inversion Convergence")

    axes[0].plot(iterations, cost, marker="o", color="tab:blue", lw=1.8)
    axes[0].set_ylabel("Cost function fk")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, grad_norm, marker="s", color="tab:orange", lw=1.6)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Gradient norm ||gk||")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    fig.savefig(args.output_file, dpi=180)
    plt.close(fig)
    print(f"Saved convergence plot to {args.output_file}")


if __name__ == "__main__":
    main()
