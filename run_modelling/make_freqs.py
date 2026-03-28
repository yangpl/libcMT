#!/usr/bin/env python3
"""Create an HDF5 frequency file for the run_modelling example."""
import h5py
import numpy as np

#FREQS = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float32)
FREQS = np.array([1000.0], dtype=np.float32)

with h5py.File("freqs.h5", "w") as f:
    f.create_dataset("freqs", data=FREQS)
