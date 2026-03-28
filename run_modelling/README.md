# `run_modelling`

This directory is a runnable template for 3D frequency-domain MT forward modelling with the `libcMT` geometric multigrid solver in the parent directory.

## What This Case Represents

The shipped template is a simple layered MT benchmark:

- Nonuniform input mesh stored in HDF5, with dimensions `160 x 160 x 100`
- Near-center cell sizes: `dx=125 m`, `dy=125 m`, `dz=40 m`
- Physical domain: `x,y in [-10 km, 10 km]`, `z in [0, 4 km]`
- 101 receivers along the x-axis at `z=1000 m`
- Frequencies: `0.001, 0.01, 0.1, 1, 10, 100, 1000 Hz`
- Output responses: `Zxx`, `Zxy`, `Zyx`, `Zyy`

Relevant files:

- `inputpar.txt`: runtime parameters
- `receivers.h5`: receiver locations and orientations in HDF5 format
- `model.h5`: mesh and resistivity model in HDF5 format

## Files

- `run.sh`: regenerates `model.h5` and `receivers.h5`, rewrites `inputpar.txt`, and launches the solver.
- `create_model.py`: generates the benchmark HDF5 mesh/model file.
- `create_acquisition.py`: generates the MT receiver line used by the template.
- `mt1d_reference.py`: computes the 1D MT impedance reference for the shipped layered model.
- `plot_mt_comparison.py`: compares `mt_data.h5` against the 1D MT reference.
- `visualize_mt_results.py`: plots apparent resistivity versus period for a selected station from `mt_data.h5` and also writes 2D maps of apparent resistivity and phase for each frequency when receivers form a grid.
- `plot_apparent_resistivity_phase.py`: compatibility wrapper that forwards to `visualize_mt_results.py`.

## Expected Runtime Flow

1. Build the solver in the parent directory with `make`.
2. Enter `run_modelling`.
3. Run `./run.sh`.
4. The script regenerates the HDF5 model file and HDF5 receiver file.
5. The script rewrites `inputpar.txt`.
6. `libcMT` reads the parameter list from `inputpar.txt`.
7. `libcMT` solves the XY and YX MT boundary polarizations separately with the total-field split `u_total = u_interior + u_bc`.
8. Output is written as `mt_data.h5`.
9. Run `python mt1d_reference.py` to generate `mt1d_reference.txt`.
10. Run `python plot_mt_comparison.py [mt_data_file] [reference_file] [output_prefix]` to generate MT comparison figures.
11. Run `python visualize_mt_results.py [mt_data_file] [output_prefix] --station 1` to generate the station-1 apparent-resistivity-versus-period curve and per-frequency 2D apparent-resistivity/phase maps.
12. If you still use the older filename, `python plot_apparent_resistivity_phase.py [mt_data_file] [output_prefix]` runs the same visualization code.

## Input Contract

`libcMT` requires the following groups of inputs.

### Grid And Model

- `fmodel`: HDF5 file containing `fx1`, `fx2`, `fx3`, `frho11`, `frho22`, `frho33`

For this template the three resistivity tensors are generated as identical isotropic datasets in `model.h5`.

### Survey Geometry

- `frec`: HDF5 receiver file with datasets `receiver_position`, `receiver_azimuth`, `receiver_dip`, `receiver_index`

### Multigrid Mesh And Solver Controls

- `nb`: minimum number of padding cells added on each side of the source mesh
- `lextend`: extra padding distance beyond the source mesh on the sides and bottom; the code raises it if needed to satisfy the skin-depth rule
- `nskin`: multiplier for the minimum padding target in units of skin depth, default `4`
- `rho_skin`: resistivity used in the skin-depth estimate, default `rhomax_noair`
- `qpad`: geometric growth factor used in the padding shell
- `rho_air`: resistivity used in the air extension
- `cycleopt`: `1=V`, `2=F`, `3=W`
- `ncycle`: maximum multigrid cycles
- `v1`, `v2`: pre/post smoothing counts
- `tol`: relative residual tolerance
- `isemicoarsen`: enables semi-coarsening

The default workflow no longer needs `n1`, `n2`, or `n3`. It keeps the input mesh in the model interior, adds shell padding, rounds the final grid sizes up to powers of two, and enforces `lextend >= nskin * skin_depth(lowest_frequency, rho_skin)`.

## Model Preparation

`create_model.py` creates a layered resistivity model and writes it to `model.h5`:

- `x` and `y`: symmetric nonuniform node coordinates built from the same geometric-grid logic used in `make_model_1d.py`
- `z`: uniform node coordinates with `dz=40 m`

- `0-1000 m`: `0.3 Ohm m`
- `1000-1500 m`: `1.0 Ohm m`
- `1500-1600 m`: `100 Ohm m`
- below `1600 m`: `1000 Ohm m`

It writes:

- `fx1`, `fx2`: nonuniform node coordinates
- `fx3`: uniform node coordinates
- `frho11`, `frho22`, `frho33`: model used by `libcMT`

## Output Format

The active solver writes one HDF5 file:

- `mt_data.h5`

The file contains these datasets:

- `frequencies`: shape `(nfreq,)`
- `receiver_index`: shape `(nrec,)`, 1-based receiver IDs
- `receiver_position`: shape `(nrec, 3)` for `x`, `y`, `z`
- `Zxx`, `Zxy`, `Zyx`, `Zyy`: shape `(nfreq, nrec, 2)` with the last axis storing `[real, imag]`

The reference/validation scripts write:

- `mt1d_reference.txt`
- `mt_compare_0.25Hz.png`
- `mt_compare_0.75Hz.png`
- `mt_station_001_rhoa_period.png`
- `mt_map_<freq>.png`

With no command-line arguments, the plotting scripts default to:

- `plot_mt_comparison.py mt_data.h5 mt1d_reference.txt mt_compare`
- `visualize_mt_results.py mt_data.h5 mt`
- `plot_apparent_resistivity_phase.py mt_data.h5 mt`

## Important Limitations

Several inputs are present in the template or parser but are not effective in the current forward path:

- `frho_init` is present in this directory but unused by the shipped input file.

In practice, this template behaves as a 3D MT total-field benchmark with automatic nonuniform mesh generation around the receiver domain and 1D column-derived boundary fields.

For the shipped layered benchmark, the boundary-column averaging is adequate because the model is isotropic. It should be revisited if you move to full rotated anisotropy.

## Recommended Next Steps

- Keep `run.sh` only as a convenience launcher; it overwrites `inputpar.txt` on every run.
- If you want reusable cases, create one directory per survey/model variant.
- The launcher and generators are pure Python now, so the template no longer depends on a Fortran compiler.
