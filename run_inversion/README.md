# `run_inversion`

This directory is a runnable inversion template for the `libcMT` solver in the parent directory. It mirrors the structure of `run_modelling`, but adds the extra assets needed for a gradient-output smoke test:

- a **synthetic true model** used to generate observed data
- an **initial model** used by the inversion run
- a launcher that runs `mode=2` to write `gradient_iter0000.h5`
- helper scripts to save TensorMesh-based 3D figures, plot convergence, and reconstruct `model_recovered.h5` from an optimizer log

## What the shipped example does

The default case now uses the same mesh and layered background as
`run_modelling/make_model_3d.py`:

- input mesh: `100 x 100 x 100` nodes, or `99 x 99 x 100` cells
- near-center cell sizes: `dx=100 m`, `dy=100 m`, `dz=50 m`
- physical domain: `x,y in [-10 km, 10 km]`, `z in [0, 5 km]`
- background model: layered anisotropic resistivity with interfaces at `1000 m`, `2300 m`, and `2400 m`
- true model: that background plus a `10 Ohm m` conductive block near the center
- receivers: `3 x 3` grid at `z=1000 m`
- frequency list: default `1 Hz`
- default initial model: **layered background** when `INITIAL_MODE=background`, or the true model when `copy_true` is requested

The inversion path is still sensitive to the starting model and line search. So the default launcher uses `mode=2` and remains better suited to workflow validation than to an aggressive recovery benchmark. It verifies that:

1. the true and initial HDF5 models can be generated
2. the receiver file can be generated
3. `mode=2` reads the initial model, receiver file, and data file
4. the model and gradient HDF5 outputs can be plotted as TensorMesh-based 3D figures

## Files

- `run.sh`: builds the inputs, runs `mode=2`, and saves 3D model and gradient figures.
- `make_models_3d.py`: writes `model_true.h5` and `model_init.h5`.
- `make_acquisition.py`: writes `receivers.h5` and `receivers_layout.png`.
- `plot_iterate.py`: turns `iterate.txt` into `inversion_convergence.png`.
- `extract_final_model.py`: parses `inversion.log` and writes `model_recovered.h5`.
- `plot_models_3d.py`: saves coarsened TensorMesh 3D voxel figures and TensorMesh slicer PNGs for the true and initial models.
- `plot_gradient_3d.py`: saves TensorMesh 3D slicer figures for `gradient_iterXXXX.h5` files written by inversion or `mode=2`.

## Default workflow

1. Build the solver in the parent directory with `make`.
2. Enter `run_inversion`.
3. Run `./run.sh`.
4. The script writes `model_true.h5`, `model_init.h5`, and `receivers.h5`.
5. It saves `model_3d_true.png`, `model_3d_initial.png`, `model_3d_slicer_true.png`, and `model_3d_slicer_initial.png`.
6. It launches `libcMT` with `mode=2` to write `gradient_iter0000.h5`.
7. It writes:
   - `gradient_iter0000.h5`
   - `gradient_iter0000_grad_mh.png`
   - `gradient_iter0000_grad_mv.png`

## Full Inversion Notes

`run_modelling` only needs the multigrid tolerance `tol`.  For inversion, there are two different tolerances that are useful in practice:

- `tol`: multigrid solver tolerance
- `gtol`: optimizer stopping tolerance

A small backward-compatible change was added in `do_inversion.c` so inversion now accepts `gtol=` and falls back to `tol=` if `gtol` is not provided.  This is relevant when you switch `run.sh` from the default gradient-only `mode=2` back to a full `mode=1` inversion run.

## Default inversion command

`run.sh` runs the equivalent of:

```bash
../bin/libcMT \
    mode=2 \
    freqs=1 \
    fmodel=model_init.h5 \
    frec=receivers.h5 \
    fdata=mt_data.h5
```

For a full inversion smoke test, a loose `gtol` can stop after the initial objective/gradient evaluation instead of driving a potentially fragile line search.

## Output contract

The generated files use the same HDF5 conventions as the forward template:

- `model_true.h5`, `model_init.h5`, `model_recovered.h5` contain:
  - `fx1`, `fx2`, `fx3`
  - `frho11`, `frho22`, `frho33`
- `receivers.h5` contains:
  - `receiver_position`, `receiver_azimuth`, `receiver_dip`, `receiver_index`
- `mt_data.h5` contains:
  - `frequencies`
  - `receiver_position`, `receiver_index`
  - `Zxx`, `Zxy`, `Zyx`, `Zyy`
- `gradient_iterXXXX.h5` contains:
  - `fx1`, `fx2`, `fx3`
  - `grad_mh`, `grad_mv`

Model snapshots are saved as coarsened 3D voxel figures, and gradient snapshots are saved as TensorMesh 3D slicer figures, with:

```bash
python3 plot_models_3d.py --save-prefix model_3d
python3 plot_gradient_3d.py gradient_iter0000.h5 --component grad_mh
python3 plot_gradient_3d.py gradient_iter0000.h5 --component grad_mv
```

Add `--show` to any of those commands when you also want an interactive window.

## Notes

- The inversion parameterization is VTI log-conductivity: one horizontal parameter for `sigma11 = sigma22` and one vertical parameter for `sigma33` per cell.
- `extract_final_model.py` assumes the `final x:` line is present in `inversion.log`, which requires `verb=1` during inversion.
- If you increase the model size or frequency count, runtime will grow quickly because each objective evaluation performs forward and adjoint solves.
