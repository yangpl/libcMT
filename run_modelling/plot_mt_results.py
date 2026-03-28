import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("Zxx", "Zxy", "Zyx", "Zyy")
MU0 = 4.0e-7 * np.pi
COMPONENT_COLORS = {
    "Zxx": "tab:blue",
    "Zxy": "tab:green",
    "Zyx": "tab:red",
    "Zyy": "tab:orange",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot MT apparent resistivity and phase versus period for selected stations."
    )
    parser.add_argument(
        "mt_data_file",
        nargs="?",
        default="mt_data.h5",
        help="Path to the MT HDF5 file. Default: mt_data.h5",
    )
    parser.add_argument(
        "output_prefix",
        nargs="?",
        default="mt",
        help="Prefix for output figure names. Default: mt",
    )
    parser.add_argument(
        "--station",
        type=int,
        help="1-based station index to plot. By default, plots all stations.",
    )
    return parser.parse_args()


def read_mt_data(path):
    with h5py.File(path, "r") as handle:
        frequencies = np.asarray(handle["frequencies"][:], dtype=float)
        receiver_index = np.asarray(handle["receiver_index"][:], dtype=int)
        receiver_position = np.asarray(handle["receiver_position"][:], dtype=float)

        data = {
            "frequencies": frequencies,
            "receiver_index": receiver_index,
            "receiver_position": receiver_position,
        }
        for component in COMPONENTS:
            values = np.asarray(handle[component][:], dtype=float)
            data[component] = values[..., 0] + 1j * values[..., 1]

    return data


def apparent_resistivity(values, frequency):
    omega = 2.0 * np.pi * frequency
    return np.abs(values) ** 2 / (MU0 * omega)


def phase_degrees(values):
    return np.rad2deg(np.angle(values))


def periods_from_frequencies(frequencies):
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = 1.0 / np.asarray(frequencies, dtype=float)
    return periods


def station_zero_based(receiver_index, station):
    matches = np.where(receiver_index == station)[0]
    if matches.size == 0:
        raise ValueError(
            f"station {station} not found; available stations are {receiver_index.min()} to {receiver_index.max()}"
        )
    return int(matches[0])


def ordered_station_series(data, station):
    istation = station_zero_based(data["receiver_index"], station)
    periods = periods_from_frequencies(data["frequencies"])
    order = np.argsort(periods, kind="stable")
    return istation, periods[order], data["frequencies"][order], data["receiver_position"][istation], order


def save_station_period_plot(data, station, output_prefix):
    istation, periods, frequencies, position, order = ordered_station_series(data, station)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 9.0), sharex=True, constrained_layout=True)
    ax_rhoa, ax_phase = axes

    for component in COMPONENTS:
        values = data[component][order, istation]
        rhoa = apparent_resistivity(values, frequencies)
        phase = phase_degrees(values)
        color = COMPONENT_COLORS[component]

        ax_rhoa.plot(periods, rhoa, marker="o", ms=4, lw=1.8, color=color, label=component)
        ax_phase.plot(periods, phase, marker="o", ms=4, lw=1.8, color=color, label=component)

    ax_rhoa.set_xscale("log")
    ax_rhoa.set_yscale("log")
    ax_rhoa.set_ylabel("Apparent Resistivity (Ohm m)")
    ax_rhoa.set_title(
        f"Station {station} MT Response vs Period\n"
        f"x={position[0]:g} m, y={position[1]:g} m, z={position[2]:g} m"
    )
    ax_rhoa.grid(True, which="both", alpha=0.3)
    ax_rhoa.legend()

    ax_phase.set_xscale("log")
    ax_phase.set_xlabel("Period (s)")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.grid(True, which="both", alpha=0.3)

    fig.savefig(f"{output_prefix}_station_{station:03d}_mt_period.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    data = read_mt_data(Path(args.mt_data_file))

    if args.station is None:
        for station in data["receiver_index"]:
            save_station_period_plot(data, int(station), args.output_prefix)
    else:
        save_station_period_plot(data, args.station, args.output_prefix)


if __name__ == "__main__":
    main()
