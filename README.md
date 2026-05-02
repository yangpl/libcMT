# `libcMT`

`libcMT` is a C/MPI code for 3D frequency-domain magnetotelluric (MT) forward modelling and inversion on a nonuniform mesh using a geometric multigrid solver.

## Repository Structure

- `src/`: solver sources and the build recipe.
- `include/`: public headers for shared data structures and utilities.
- `bin/`: build output. The executable is `bin/libcMT`.
- `run_modelling/`: runnable example that generates HDF5 inputs, runs the solver, and plots MT responses.
- `run_inversion/`: runnable inversion template with synthetic true/initial models, receiver generation, inversion/gradient launch scripts, and plotting helpers.
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
- `src/inject_extract.c`: receiver extraction and adjoint-source injection.
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
- HDF5 compiler and linker flags are resolved with `pkg-config`.

## Runtime Interface

The executable accepts command-line arguments as `key=value`.

Important arguments used by the current code:

- `mode=0` for forward modelling, `mode=1` for full inversion, `mode=2` for a single objective/gradient evaluation.
- `freqs=...` for a comma-separated frequency list, or `ffreqs=...` for an HDF5 file containing dataset `freqs`.
- `fmodel=...` for the HDF5 model file.
- `frec=...` for the HDF5 receiver file.
- `fdata=...` for the forward output file in modelling mode, or observed MT data in inversion/gradient mode. If omitted in modelling mode, the output file is `mt_data.h5`.
- `verb=0|1` for verbosity.
- `tol`, `cycleopt`, `ncycle`, `v1`, `v2`, `isemicoarsen` for multigrid control.
- `nb`, `rho_skin`, `rho_air` for automatic model extension and air handling.
- `niter`, `nls`, `npair`, `bound`, `method`, `ncg`, `c1`, `c2`, `alpha`, `gtol` for inversion/optimizer control. If `gtol` is not supplied, inversion falls back to `tol`.

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

Forward modelling writes `mt_data.h5`, or the path supplied by `fdata=`, in the current working directory with:

- `frequencies`
- `receiver_index`
- `receiver_position`
- `Zxx`, `Zxy`, `Zyx`, `Zyy`

In inversion mode, the optimizer also writes `iterate.txt`, per-iteration model snapshots named `model_iterXXXX.h5`, and per-iteration gradients named `gradient_iterXXXX.h5`. In `mode=2`, the code writes `gradient_iter0000.h5` after the first objective/gradient evaluation and exits without running a line search.

## Forward Modelling Template

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

Additional plotting can be run with `plot_mt_results.py` to write station-period figures and per-frequency receiver maps.

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

## Inversion Template

The `run_inversion/` directory provides a small end-to-end template for synthetic inversion experiments.

Files there:

- `run_inversion/run.sh`: regenerates the model and receiver inputs, plots true/initial models, and launches the solver in inversion mode.
- `run_inversion/make_models_3d.py`: creates `model_true.h5` and `model_init.h5`.
- `run_inversion/make_acquisition.py`: creates a `3 x 3` receiver grid in `receivers.h5` and `receivers_layout.png`.
- `run_inversion/plot_models_3d.py`: visualizes true, initial, and recovered model HDF5 files.
- `run_inversion/plot_gradient_3d.py`: visualizes `gradient_iterXXXX.h5`.
- `run_inversion/plot_iterate.py`: converts `iterate.txt` to `inversion_convergence.png`.
- `run_inversion/extract_final_model.py`: reconstructs `model_recovered.h5` from a verbose inversion log.

Typical inversion setup:

```bash
cd src
make

cd ../run_inversion
python3 make_models_3d.py
python3 make_acquisition.py

# Generate synthetic observed data from the true model.
../bin/libcMT mode=0 freqs=1 fmodel=model_true.h5 frec=receivers.h5 fdata=mt_data.h5

# Run inversion from the initial model.
../bin/libcMT mode=1 freqs=1 fmodel=model_init.h5 frec=receivers.h5 fdata=mt_data.h5 niter=20 npair=5 gtol=1e-6
```

For a faster gradient-output smoke test, use:

```bash
../bin/libcMT mode=2 freqs=1 fmodel=model_init.h5 frec=receivers.h5 fdata=mt_data.h5
```

The inversion template uses the same HDF5 model contract as the forward template. `model_iterXXXX.h5` snapshots are also reusable as `fmodel=` inputs because they include `fx1`, `fx2`, `fx3`, `frho11`, `frho22`, and `frho33`.

## Current Behavior Notes

- MPI parallelization is implemented across frequencies.
- In multi-rank forward modelling, rank 0 schedules work and collects results; worker ranks solve one frequency at a time.
- In inversion mode, rank 0 owns the optimizer and workers handle per-frequency forward/adjoint solves.
- The inversion scheduler is pipelined. A worker first solves the forward problem for one frequency and returns the sampled receiver fields. Rank 0 computes the impedance residual and adjoint receiver sources, sends those sources back to the same worker, and the worker then solves the adjoint problem and returns that frequency's gradient contribution.
- Rank 0 handles forward completions and gradient completions as independent MPI events. It can therefore service completed forward solves from other workers while some workers are still running adjoint solves, instead of blocking on one worker's gradient before receiving more forward results.
- The adjoint solve for a frequency still depends on that frequency's forward result. The implementation keeps the forward field history local to the worker that computed it, avoiding a full all-forward-then-all-adjoint phase that would require large field transfers, extra memory, or forward recomputation.
- In MPI inversion, each worker stores only one local forward/adjoint field-history slot and reuses it for each assigned frequency. This reduces worker memory by roughly a factor of `nfreq` compared with storing all frequency histories. Serial inversion still stores all frequency histories because it runs all forward solves before the adjoint pass.
- The current inversion parameterization is VTI in log-conductivity: one horizontal parameter for `sigma11 = sigma22`, and one vertical parameter for `sigma33`.
- The code treats cells with resistivity greater than or equal to `rho_air` as air when applying inversion bounds and building the extended model.
- Frequencies can be supplied directly with `freqs=...` or through `ffreqs=...` (HDF5 frequency file 'ffreqs' must contain dataset `freqs`).

## Notes On Included Test Directories

- `test_mt1d/` contains standalone 1D MT derivations, small drivers, and plotting helpers.
