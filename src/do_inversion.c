/* Inversion driver that configures and launches the optimizer.
 * Reads optimization parameters, allocates the optimizer workspace, and
 * evaluates the MT objective through the inversion callback.
 *----------------------------------------------------------------------
 *   Copyright (c) Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *----------------------------------------------------------------------
 */
#include "cstd.h"
#include "emf.h"
#include "acq.h"
#include "optim.h"
#include <mpi.h>

void read_mt_data(acq_t *acq, emf_t *emf, char *fname);
void inversion_init(acq_t *acq, emf_t *emf);
void inversion_free(emf_t *emf);
void inversion_worker_loop(acq_t *acq, emf_t *emf);
void inversion_mpi_stop(void);
void inversion_init_data_weights(acq_t *acq, emf_t *emf);
float inversion_grad(const float *x, float *g);

/* Configure the optimizer, map the starting model into log-conductivity space, and launch inversion. */
/* Rank 0 owns optimizer setup and objective evaluations. When MPI is enabled, other ranks
 * bypass the optimizer entirely and stay inside inversion_worker_loop(), waiting for model
 * broadcasts and per-frequency work assignments from the master process. */
int do_inversion(acq_t *acq, emf_t *emf)
{
  int i, j, k;
  int id;
  int ncell;
  int rank, size;
  char *fdata;
  optim_t opt;
  int status = EXIT_SUCCESS;
  float sigma_h, sigma_v;
  float sigma_min, sigma_max;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  memset(&opt, 0, sizeof(opt));
  if(!getparint("verb", &opt.verb)) opt.verb = 1;
  if(!getparint("niter", &opt.niter)) opt.niter = 100;
  if(!getparint("nls", &opt.nls)) opt.nls = 20;
  if(!getparint("npair", &opt.npair)) opt.npair = 5;
  if(!getparint("bound", &opt.bound)) opt.bound = 1;
  if(!getparint("method", &opt.method)) opt.method = 1;
  if(!getparint("ncg", &opt.ncg)) opt.ncg = 5;
  if(!getparfloat("gtol", &opt.tol)) {
    if(!getparfloat("tol", &opt.tol)) opt.tol = 1e-6f;
  }
  if(!getparfloat("c1", &opt.c1)) opt.c1 = 1e-4f;
  if(!getparfloat("c2", &opt.c2)) opt.c2 = 0.9f;
  if(!getparfloat("alpha", &opt.alpha0)) opt.alpha0 = 1.0f;
  opt.alpha = opt.alpha0;

  if(!getparstring("fdata", &fdata)) err("Need fdata=");

  ncell = emf->nx * emf->ny * emf->nz;

  /* Allocate rank-specific inversion workspaces before splitting master and workers. */
  inversion_init(acq, emf);

  if(rank != 0) {
    /* Worker ranks stay in the service loop until rank 0 broadcasts INV_CMD_STOP. */
    inversion_worker_loop(acq, emf);
    inversion_free(emf);
    return EXIT_SUCCESS;
  }

  opt.n = 2 * ncell;
  if(!optim_init(&opt, opt.n)) {
    fprintf(stderr, "failed to initialize optimizer\n");
    /* Tell workers there are no more optimization steps before tearing down shared state. */
    if(size > 1) inversion_mpi_stop();
    inversion_free(emf);
    return EXIT_FAILURE;
  }

  /* Bounds are applied in conductivity space, then converted to log-conductivity. */
  sigma_min = 1.0f / emf->rhomax_noair;
  sigma_max = 1.0f / emf->rhomin;

  for(k = 0; k < emf->nz; ++k) {
    for(j = 0; j < emf->ny; ++j) {
      for(i = 0; i < emf->nx; ++i) {
        id = i + emf->nx * (j + emf->ny * k);

        sigma_h = sqrtf((1.0f / emf->rho11[k][j][i]) * (1.0f / emf->rho22[k][j][i]));
        sigma_v = 1.0f / emf->rho33[k][j][i];
        opt.x[id] = logf(sigma_h);
        opt.x[id + ncell] = logf(sigma_v);

        /* Air cells remain fixed; subsurface cells use the global conductivity bounds. */
        if(opt.bound) {
          if(emf->rho11[k][j][i] >= emf->rho_air || emf->rho22[k][j][i] >= emf->rho_air) {
            opt.xmin[id] = opt.x[id];
            opt.xmax[id] = opt.x[id];
          } else {
            opt.xmin[id] = logf(sigma_min);
            opt.xmax[id] = logf(sigma_max);
          }

          if(emf->rho33[k][j][i] >= emf->rho_air) {
            opt.xmin[id + ncell] = opt.x[id + ncell];
            opt.xmax[id + ncell] = opt.x[id + ncell];
          } else {
            opt.xmin[id + ncell] = logf(sigma_min);
            opt.xmax[id + ncell] = logf(sigma_max);
          }
        }
      }
    }
  }

  /* Only rank 0 needs the observed MT tensors because it is the only rank that forms
   * impedance residuals and adjoint sources. */
  read_mt_data(acq, emf, fdata);
  inversion_init_data_weights(acq, emf);
  status = optim_run(&opt, inversion_grad, NULL);

  /* Tell workers there are no more optimization steps before tearing down shared state. */
  if(size > 1) inversion_mpi_stop();
  inversion_free(emf);
  optim_free(&opt);

  if(status == OPTIM_STATUS_LINE_SEARCH_FAILED) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
