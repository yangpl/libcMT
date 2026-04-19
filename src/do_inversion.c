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
void inversion_init_data_weights(acq_t *acq, emf_t *emf);
float inversion_grad(const float *x, float *g);
void write_inversion_model_hdf5(emf_t *emf, const float *x, int iter);

static int count_free_parameters(const optim_t *opt)
{
  int i, nfree = 0;

  for(i = 0; i < opt->n; ++i) {
    if(opt->xmax[i] > opt->xmin[i]) ++nfree;
  }

  return nfree;
}

static void build_gradient_check_direction(const optim_t *opt, float *p, int idir)
{
  int i;
  int target;
  int free_seen = 0;

  memset(p, 0, opt->n * sizeof(float));

  target = idir;
  for(i = 0; i < opt->n; ++i) {
    if(opt->xmax[i] <= opt->xmin[i]) continue;
    if(free_seen == target) {
      p[i] = 1.0f;
      return;
    }
    ++free_seen;
  }
}

static int run_gradient_check(const optim_t *opt)
{
  FILE *fp = NULL;
  int i, idir;
  int ncheck;
  int nfree;
  int seed;
  float eps;
  float *g = NULL;
  float *xp = NULL;
  float *xm = NULL;
  float *p = NULL;
  float *g_dummy = NULL;
  double f0, fp_cost, fm_cost;
  double fd_dir, adj_dir, abs_err, rel_err, denom;
  double max_abs_err = 0.0, max_rel_err = 0.0;

  if(!getparfloat("gradcheck_eps", &eps)) eps = 1e-4f;
  if(!getparint("gradcheck_n", &ncheck)) ncheck = 4;
  if(!getparint("gradcheck_seed", &seed)) seed = 0;

  if(eps <= 0.0f) err("gradcheck_eps must be positive");

  nfree = count_free_parameters(opt);
  if(nfree <= 0) {
    printf("Gradient check skipped: no free inversion parameters.\n");
    return EXIT_SUCCESS;
  }
  if(ncheck <= 0) ncheck = 1;
  ncheck = MIN(ncheck, nfree);

  g = alloc1float(opt->n);
  xp = alloc1float(opt->n);
  xm = alloc1float(opt->n);
  p = alloc1float(opt->n);
  g_dummy = alloc1float(opt->n);
  if(g == NULL || xp == NULL || xm == NULL || p == NULL || g_dummy == NULL)
    err("failed to allocate gradient-check buffers");

  srand((unsigned int)seed);
  f0 = inversion_grad(opt->x, g);

  printf("======= gradient check =======\n");
  printf("f(x0)=%e ||g(x0)||=%e nfree=%d eps=%g ncheck=%d\n",
         f0, l2norm(opt->n, g), nfree, eps, ncheck);

  fp = fopen("gradcheck.txt", "w");
  if(fp) {
    setvbuf(fp, NULL, _IOLBF, 0);
    fprintf(fp, "# idir fd_dot adj_dot abs_err rel_err\n");
  }

  for(idir = 0; idir < ncheck; ++idir) {
    int free_index = (seed == 0) ? idir : rand() % nfree;

    build_gradient_check_direction(opt, p, free_index);
    memcpy(xp, opt->x, opt->n * sizeof(float));
    memcpy(xm, opt->x, opt->n * sizeof(float));
    for(i = 0; i < opt->n; ++i) {
      xp[i] += eps * p[i];
      xm[i] -= eps * p[i];
    }
    if(opt->bound) {
      boundx(xp, opt->n, opt->xmin, opt->xmax);
      boundx(xm, opt->n, opt->xmin, opt->xmax);
    }

    fp_cost = inversion_grad(xp, g_dummy);
    fm_cost = inversion_grad(xm, g_dummy);
    fd_dir = (fp_cost - fm_cost) / (2.0 * eps);
    adj_dir = dotprod(opt->n, g, p);
    abs_err = fabs(fd_dir - adj_dir);
    denom = MAX(MAX(fabs(fd_dir), fabs(adj_dir)), 1e-30);
    rel_err = abs_err / denom;
    max_abs_err = MAX(max_abs_err, abs_err);
    max_rel_err = MAX(max_rel_err, rel_err);

    printf("gradcheck idir=%d free_index=%d fd=% .6e adj=% .6e abs=% .3e rel=% .3e\n",
           idir, free_index, fd_dir, adj_dir, abs_err, rel_err);
    if(fp) {
      fprintf(fp, "%d %d %.8e %.8e %.8e %.8e\n",
              idir, free_index, fd_dir, adj_dir, abs_err, rel_err);
    }
  }

  printf("gradcheck summary: max_abs=%e max_rel=%e\n", max_abs_err, max_rel_err);

  if(fp) fclose(fp);
  free1float(g);
  free1float(xp);
  free1float(xm);
  free1float(p);
  free1float(g_dummy);

  return EXIT_SUCCESS;
}

/* Configure the optimizer, map the starting model into log-conductivity space, and launch inversion. */
/* Rank 0 owns optimizer setup and objective evaluations. When MPI is enabled, other ranks
 * bypass the optimizer entirely and stay inside inversion_worker_loop(), waiting for model
 * broadcasts and per-frequency work assignments from the master process. */
int do_inversion(acq_t *acq, emf_t *emf)
{
  FILE *fp = NULL;
  int i, j, k;
  int id;
  int ncell;
  int do_gradcheck = 0;
  int rank, size;
  char *fdata;
  optim_t opt;
  int exit_status = EXIT_SUCCESS;
  float sigma_h, sigma_v;
  float sigma_min, sigma_max;
  float beta_num, beta_den, beta;

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
  /* The inversion objective is quite sensitive to the first trial step. Keep
   * the default conservative so the stock launcher does not immediately drive
   * the line search into failure from a rough starting model. */
  if(!getparfloat("alpha", &opt.alpha0)) opt.alpha0 = 0.1f;
  opt.alpha = opt.alpha0;
  getparint("gradcheck", &do_gradcheck);

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
    exit_status = EXIT_FAILURE;
    goto cleanup;
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

  if(do_gradcheck) {
    exit_status = run_gradient_check(&opt);
    goto cleanup;
  }

  opt.igrad = opt.iter = opt.ils = opt.kpair = opt.ls_fail = 0;
  opt.status = OPTIM_STATUS_RUNNING;
  opt.alpha = (opt.alpha0 > 0.0f) ? opt.alpha0 : 1.0f;
  opt.fk = opt.f0 = inversion_grad(opt.x, opt.g);
  opt.igrad = 1;
  opt.gk_norm = opt.g0_norm = l2norm(opt.n, opt.g);

  if(opt.verb) {
    printf("======= optimization starts ===========\n");
    fp = fopen("iterate.txt", "w");
    if(fp) {
      setvbuf(fp, NULL, _IOLBF, 0);
      fprintf(fp, "================================================================================\n");
      fprintf(fp, "method: %s\n", optim_method_name(opt.method));
      fprintf(fp, "%6s %14s %14s %14s %10s %6s %8s\n",
              "iter", "fk", "fk/f0", "||gk||", "alpha", "nls", "ngrad");
      fprintf(fp, "================================================================================\n");
      fflush(fp);
    }
  }

  for(opt.iter = 0; opt.iter < opt.niter; ++opt.iter) {
    opt.gk_norm = l2norm(opt.n, opt.g);
    if(opt.verb) {
      printf("iteration=%d fk=%g ||g||=%g\n", opt.iter, opt.fk, opt.gk_norm);
      if(fp) {
        fprintf(fp, "%6d %14.6e %14.6e %14.6e %10.4e %6d %8d\n",
                opt.iter, opt.fk, opt.fk / opt.f0, opt.gk_norm,
                opt.alpha, opt.ils, opt.igrad);
        fflush(fp);
      }
    }

    write_inversion_model_hdf5(emf, opt.x, opt.iter);

    if(opt.gk_norm <= opt.tol * MAX(1.0f, opt.g0_norm)) {
      opt.status = OPTIM_STATUS_CONVERGED;
      break;
    }

    switch(opt.method) {
      case OPTIM_METHOD_NEWTON_CG:
        cg_solve(opt.n, opt.x, opt.g, opt.d, NULL, &opt);
        break;
      case OPTIM_METHOD_NLCG:
        if(opt.iter == 0) {
          flipsign(opt.n, opt.g, opt.d);
        } else {
          beta_num = dotprod(opt.n, opt.g, opt.g);
          beta_den = dotprod(opt.n, opt.g_prev, opt.g_prev);
          beta = (beta_den > 0.0f) ? beta_num / beta_den : 0.0f;
          for(i = 0; i < opt.n; ++i) opt.d[i] = -opt.g[i] + beta * opt.d[i];
        }
        memcpy(opt.g_prev, opt.g, opt.n * sizeof(float));
        break;
      case OPTIM_METHOD_LBFGS:
      default:
        if(opt.iter == 0) {
          flipsign(opt.n, opt.g, opt.d);
        } else {
          lbfgs_update(opt.n, opt.x, opt.g, opt.sk, opt.yk, &opt);
          lbfgs_descent(opt.n, opt.g, opt.d, opt.sk, opt.yk, &opt);
        }
        lbfgs_save(opt.n, opt.x, opt.g, opt.sk, opt.yk, &opt);
        break;
    }
    line_search(opt.n, opt.x, opt.g, opt.d, inversion_grad, &opt);
    if(opt.ls_fail) {
      opt.status = OPTIM_STATUS_LINE_SEARCH_FAILED;
      break;
    }
  }

  if(opt.status == OPTIM_STATUS_RUNNING) {
    opt.status = (opt.iter >= opt.niter) ? OPTIM_STATUS_MAX_ITER : OPTIM_STATUS_CONVERGED;
  }

  if(fp) {
    switch(opt.status) {
      case OPTIM_STATUS_CONVERGED:
        fprintf(fp, "==> Convergence reached.\n");
        break;
      case OPTIM_STATUS_MAX_ITER:
        fprintf(fp, "==> Maximum iteration number reached.\n");
        break;
      case OPTIM_STATUS_LINE_SEARCH_FAILED:
        fprintf(fp, "==> Line search failed.\n");
        break;
      default:
        break;
    }
    fflush(fp);
    fclose(fp);
  }
  exit_status = (opt.status == OPTIM_STATUS_LINE_SEARCH_FAILED) ? EXIT_FAILURE : EXIT_SUCCESS;

cleanup:
  /* Tell workers there are no more optimization steps before tearing down shared state. */
  if(size > 1) {
    int command = 2;
    MPI_Bcast(&command, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  inversion_free(emf);
  optim_free(&opt);
  return exit_status;
}
