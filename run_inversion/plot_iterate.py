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
    fk_over_f0: list[float] = []
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
            fk_over_f0.append(float(parts[2]))
            grad_norm.append(float(parts[3]))
        except ValueError:
            continue

    if not iterations:
        raise ValueError(f"No iteration rows found in {path}")
    return iterations, fk_over_f0, grad_norm


def main() -> None:
    args = parse_args()
    iterations, fk_over_f0, grad_norm = read_iterates(Path(args.iterate_file))

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax1.plot(iterations, fk_over_f0, marker="o", lw=1.8)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("fk / f0")
    ax1.set_title("Inversion Convergence")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(iterations, grad_norm, marker="s", lw=1.4)
    ax2.set_ylabel("||gk||")
    ax2.set_yscale("log")

    fig.savefig(args.output_file, dpi=180)
    plt.close(fig)
    print(f"Saved convergence plot to {args.output_file}")


if __name__ == "__main__":
    main()
