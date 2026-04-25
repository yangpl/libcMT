# `libcMT`

`libcMT` is a C/MPI code for 3D frequency-domain magnetotelluric (MT) forward modelling and inversion on a nonuniform mesh using a geometric multigrid solver.

## Repository Structure

- `src/`: solver sources and the build recipe.
- `include/`: public headers for shared data structures and utilities.
- `bin/`: build output. The executable is `bin/libcMT`.
- `run_modelling/`: runnable example that generates HDF5 inputs, runs the solver, and plots MT responses.
- `test_mt1d/`: standalone 1D MT reference experiments and notes.
- `test_parallelization/`: small MPI master/worker scheduling example.
- `doc/`: MT boundary-condition notes and PDFs.

## Main Source Files

- `src/main.c`: initializes MPI, parses `key=value` arguments, and dispatches modelling or inversion.
- `src/do_modelling.c`: forward solve driver, including MPI frequency scheduling.
- `src/do_inversion.c`: optimizer setup and inversion entry point.
- `src/inversion.c`: forward/adjoint objective and gradient evaluation.
- `src/gmg.c`: geometric multigrid hierarchy, cycles, smoothers, and residual handling.
- `src/extend_model.c`: extends the input model with padding cells for the computational grid.
- `src/mt1d_solve.c`: 1D MT boundary-field solver used to seed the 3D solve.
- `src/extract_inject.c`: receiver extraction and adjoint-source injection.
- `src/emf_init_free.c`: reads frequencies and the resistivity model from HDF5.
- `src/acq_init_free.c`: reads receiver geometry from HDF5.
- `src/read_write.c`: reads observed MT data and writes modelled MT responses.
- `src/optim.c`: nonlinear optimization routines used by inversion.
- `src/cstd.c`: allocation helpers, parser utilities, and shared support code.

## Build

Build from `src/`:

```bash
cd src
make
```

This produces `../bin/libcMT`.

Clean the build with:

```bash
cd src
make clean
```

Notes:

- The Makefile uses `mpicc`.
- HDF5 include/library paths are currently set in `src/Makefile`.

## Runtime Interface

The executable accepts command-line arguments as `key=value`.

Important arguments used by the current code:

- `mode=0` for forward modelling, `mode=1` for inversion.
- `freqs=...` for a comma-separated frequency list, or `ffreqs=...` for an HDF5 file containing dataset `freqs`.
- `fmodel=...` for the HDF5 model file.
- `frec=...` for the HDF5 receiver file.
- `fdata=...` for observed MT data in inversion mode.
- `verb=0|1` for verbosity.
- `tol`, `cycleopt`, `ncycle`, `v1`, `v2`, `isemicoarsen` for multigrid control.
- `nb`, `rho_skin`, `rho_air` for automatic model extension and air handling.
- `niter`, `nls`, `npair`, `bound`, `method`, `ncg`, `c1`, `c2`, `alpha` for inversion/optimizer control.

## Input Files

### Model file `fmodel`

The solver expects an HDF5 file with:

- `fx1`, `fx2`, `fx3`: node coordinates.
- `frho11`, `frho22`, `frho33`: resistivity tensors with shape `[nz, ny, nx]`.

### Receiver file `frec`

The solver expects an HDF5 file with:

- `receiver_position`: shape `[nrec, 3]`.
- `receiver_azimuth`: shape `[nrec]`.
- `receiver_dip`: shape `[nrec]`.
- `receiver_index`: written by the example generators, though the solver only reads the first three datasets above.

### Frequency file `ffreqs`

If frequencies are supplied from HDF5, the file must contain:

- `freqs`: shape `[nfreq]`.

### Observed data file `fdata`

For inversion, the observed MT data file must contain:

- `Zxx`, `Zxy`, `Zyx`, `Zyy`: shape `[nfreq, nrec, 2]`, with real and imaginary parts in the last axis.

## Output

Forward modelling writes `mt_data.h5` in the current working directory with:

- `frequencies`
- `receiver_index`
- `receiver_position`
- `Zxx`, `Zxy`, `Zyx`, `Zyy`

In inversion mode, the optimizer also writes `iterate.txt`.

## Example Workflow

The current runnable example lives in `run_modelling/`.

Files there:

- `run_modelling/run.sh`: regenerates inputs and runs a forward model.
- `run_modelling/make_model_3d.py`: creates `model.h5`.
- `run_modelling/make_acquisition.py`: creates `receivers.h5` and `receivers_layout.png`.
- `run_modelling/make_freqs.py`: creates `freqs.h5`.
- `run_modelling/plot_mt_results.py`: plots apparent resistivity/phase from `mt_data.h5`.
- `run_modelling/plot_model_3d.py`: visualizes the model.

Typical run:

```bash
cd src
make

cd ../run_modelling
./run.sh
```

The shipped `run.sh` currently:

- regenerates `model.h5`, `receivers.h5`, and `freqs.h5`
- runs `../bin/libcMT` with inline `freqs=0.01,0.1,1,10,100`
- writes `mt_data.h5` in `run_modelling/`

Manual forward examples:

```bash
./bin/libcMT freqs=0.01,0.1,1,10,100 fmodel=run_modelling/model.h5 frec=run_modelling/receivers.h5
```

```bash
mpirun -np 4 ./bin/libcMT ffreqs=run_modelling/freqs.h5 fmodel=run_modelling/model.h5 frec=run_modelling/receivers.h5
```

Manual inversion example:

```bash
./bin/libcMT mode=1 freqs=0.01,0.1 fmodel=model.h5 frec=receivers.h5 fdata=mt_data.h5 niter=20 npair=5
```

## Current Behavior Notes

- MPI parallelization is implemented across frequencies.
- In multi-rank forward modelling, rank 0 schedules work and collects results; worker ranks solve one frequency at a time.
- In inversion mode, rank 0 owns the optimizer and workers handle per-frequency forward/adjoint solves.
- The current inversion parameterization is VTI in log-conductivity: one horizontal parameter for `sigma11 = sigma22`, and one vertical parameter for `sigma33`.
- The code treats cells with resistivity greater than or equal to `rho_air` as air when applying inversion bounds and building the extended model.
- Frequencies can be supplied directly with `freqs=...` or through `ffreqs=...`; HDF5 frequency files must contain dataset `freqs`.

## Notes On Included Test Directories

- `test_mt1d/` contains standalone 1D MT derivations, small drivers, and plotting helpers.
- `test_parallelization/` contains a minimal MPI master/worker example, not the main solver.
