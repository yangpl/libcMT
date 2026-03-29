/* Objective and adjoint machinery for MT inversion.
 * Allocates inversion-side data buffers, performs forward and adjoint
 * solves, forms impedance residuals, and returns misfit/gradient values.
 *----------------------------------------------------------------------
 *   Copyright (c) Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *----------------------------------------------------------------------
 */
#include "cstd.h"
#include "emf.h"
#include "acq.h"
#include "gmg.h"
#include <mpi.h>

#define INV_CMD_EVAL 1
#define INV_CMD_STOP 2

#define TAG_INV_WORK 101
#define TAG_INV_PHASE_STOP 102
#define TAG_INV_FORWARD_INDEX 103
#define TAG_INV_FORWARD_DATA 104
#define TAG_INV_ADJOINT_SOURCES 105
#define TAG_INV_GRAD_INDEX 106
#define TAG_INV_GRAD_DATA 107

static emf_t *inv_emf;
static acq_t *inv_acq;

void extend_model_init(emf_t *emf, int ifreq);
void extend_model_free(emf_t *emf);

void mt1d_efield_at_boundary(gmg_t *gmg, double freq, int ipolar);
void extract_mt_data(acq_t *acq, int ifreq, int ipolar, float _Complex ***E);
void inject_adjoint_sources(acq_t *acq, int ifreq, int ipolar);

void read_mt_data(acq_t *acq, emf_t *emf, char *fname);

static void model_from_vector(const float *x)
{
  int i, j, k;
  int id;
  int ncell = inv_emf->nx * inv_emf->ny * inv_emf->nz;

  for(k = 0; k < inv_emf->nz; ++k) {
    for(j = 0; j < inv_emf->ny; ++j) {
      for(i = 0; i < inv_emf->nx; ++i) {
        id = i + inv_emf->nx * (j + inv_emf->ny * k);
        inv_emf->rho11[k][j][i] = expf(-x[id]);
        inv_emf->rho22[k][j][i] = expf(-x[id]);
        inv_emf->rho33[k][j][i] = expf(-x[id + ncell]);
      }
    }
  }
}

static void clear_inversion_pointers(emf_t *emf)
{
  emf->d_Ex = NULL;
  emf->d_Ey = NULL;
  emf->d_Hx = NULL;
  emf->d_Hy = NULL;
  emf->obs_Zxx = NULL;
  emf->obs_Zxy = NULL;
  emf->obs_Zyx = NULL;
  emf->obs_Zyy = NULL;
  emf->cal_Zxx = NULL;
  emf->cal_Zxy = NULL;
  emf->cal_Zyx = NULL;
  emf->cal_Zyy = NULL;
  emf->res_Zxx = NULL;
  emf->res_Zxy = NULL;
  emf->res_Zyx = NULL;
  emf->res_Zyy = NULL;
  emf->w_Zxx = NULL;
  emf->w_Zxy = NULL;
  emf->w_Zyx = NULL;
  emf->w_Zyy = NULL;
  emf->s_Ex = NULL;
  emf->s_Ey = NULL;
  emf->s_Hx = NULL;
  emf->s_Hy = NULL;
  emf->Efwd = NULL;
  emf->Eadj = NULL;
}

void inversion_init_data_weights(acq_t *acq, emf_t *emf)
{
  int ifreq, irec;

  if(emf->w_Zxx == NULL || emf->w_Zxy == NULL || emf->w_Zyx == NULL || emf->w_Zyy == NULL) {
    err("inversion data weights must be allocated before initialization");
  }

  for(ifreq = 0; ifreq < emf->nfreq; ++ifreq) {
    for(irec = 0; irec < acq->nrec; ++irec) {
      float abs_zxx = cabsf(emf->obs_Zxx[ifreq][irec]);
      float abs_zxy = cabsf(emf->obs_Zxy[ifreq][irec]);
      float abs_zyx = cabsf(emf->obs_Zyx[ifreq][irec]);
      float abs_zyy = cabsf(emf->obs_Zyy[ifreq][irec]);
      float cross = 0.03f * sqrtf(abs_zxy * abs_zyx);

      emf->w_Zxy[ifreq][irec] = 1.0f / MAX(0.03f * abs_zxy, 1e-30f);
      emf->w_Zyx[ifreq][irec] = 1.0f / MAX(0.03f * abs_zyx, 1e-30f);
      emf->w_Zxx[ifreq][irec] = 1.0f / MAX(MAX(0.2f * abs_zxx, cross), 1e-30f);
      emf->w_Zyy[ifreq][irec] = 1.0f / MAX(MAX(0.2f * abs_zyy, cross), 1e-30f);
    }
  }
}

/* Rank 0 reuses these arrays across objective evaluations, so each call starts by clearing
 * the assembled receiver data, impedance residuals, and adjoint-source buffers. Workers
 * keep their own per-frequency field caches and never touch the master-owned tensors. */
static void zero_master_buffers(void)
{
  int ncell = inv_emf->nx * inv_emf->ny * inv_emf->nz;

  memset(&inv_emf->d_Ex[0][0][0], 0, 2 * inv_acq->nrec * inv_emf->nfreq * sizeof(float _Complex));
  memset(&inv_emf->d_Ey[0][0][0], 0, 2 * inv_acq->nrec * inv_emf->nfreq * sizeof(float _Complex));
  memset(&inv_emf->d_Hx[0][0][0], 0, 2 * inv_acq->nrec * inv_emf->nfreq * sizeof(float _Complex));
  memset(&inv_emf->d_Hy[0][0][0], 0, 2 * inv_acq->nrec * inv_emf->nfreq * sizeof(float _Complex));
  memset(&inv_emf->cal_Zxx[0][0], 0, (size_t)inv_emf->nfreq * inv_acq->nrec * sizeof(float _Complex));
  memset(&inv_emf->cal_Zxy[0][0], 0, (size_t)inv_emf->nfreq * inv_acq->nrec * sizeof(float _Complex));
  memset(&inv_emf->cal_Zyx[0][0], 0, (size_t)inv_emf->nfreq * inv_acq->nrec * sizeof(float _Complex));
  memset(&inv_emf->cal_Zyy[0][0], 0, (size_t)inv_emf->nfreq * inv_acq->nrec * sizeof(float _Complex));
  memset(&inv_emf->res_Zxx[0][0], 0, (size_t)inv_emf->nfreq * inv_acq->nrec * sizeof(float _Complex));
  memset(&inv_emf->res_Zxy[0][0], 0, (size_t)inv_emf->nfreq * inv_acq->nrec * sizeof(float _Complex));
  memset(&inv_emf->res_Zyx[0][0], 0, (size_t)inv_emf->nfreq * inv_acq->nrec * sizeof(float _Complex));
  memset(&inv_emf->res_Zyy[0][0], 0, (size_t)inv_emf->nfreq * inv_acq->nrec * sizeof(float _Complex));
  memset(&inv_emf->s_Ex[0][0][0], 0, 2 * inv_acq->nrec * inv_emf->nfreq * sizeof(float _Complex));
  memset(&inv_emf->s_Ey[0][0][0], 0, 2 * inv_acq->nrec * inv_emf->nfreq * sizeof(float _Complex));
  memset(&inv_emf->s_Hx[0][0][0], 0, 2 * inv_acq->nrec * inv_emf->nfreq * sizeof(float _Complex));
  memset(&inv_emf->s_Hy[0][0][0], 0, 2 * inv_acq->nrec * inv_emf->nfreq * sizeof(float _Complex));

  (void)ncell;
}

static void pack_forward_data(int ifreq, float _Complex *buffer)
{
  int irec;
  for(irec = 0; irec < inv_acq->nrec; ++irec) {
    buffer[0 * inv_acq->nrec + irec] = inv_emf->d_Ex[0][ifreq][irec];
    buffer[1 * inv_acq->nrec + irec] = inv_emf->d_Ex[1][ifreq][irec];
    buffer[2 * inv_acq->nrec + irec] = inv_emf->d_Ey[0][ifreq][irec];
    buffer[3 * inv_acq->nrec + irec] = inv_emf->d_Ey[1][ifreq][irec];
    buffer[4 * inv_acq->nrec + irec] = inv_emf->d_Hx[0][ifreq][irec];
    buffer[5 * inv_acq->nrec + irec] = inv_emf->d_Hx[1][ifreq][irec];
    buffer[6 * inv_acq->nrec + irec] = inv_emf->d_Hy[0][ifreq][irec];
    buffer[7 * inv_acq->nrec + irec] = inv_emf->d_Hy[1][ifreq][irec];
  }
}

static void unpack_forward_data(int ifreq, const float _Complex *buffer)
{
  int irec;
  for(irec = 0; irec < inv_acq->nrec; ++irec) {
    inv_emf->d_Ex[0][ifreq][irec] = buffer[0 * inv_acq->nrec + irec];
    inv_emf->d_Ex[1][ifreq][irec] = buffer[1 * inv_acq->nrec + irec];
    inv_emf->d_Ey[0][ifreq][irec] = buffer[2 * inv_acq->nrec + irec];
    inv_emf->d_Ey[1][ifreq][irec] = buffer[3 * inv_acq->nrec + irec];
    inv_emf->d_Hx[0][ifreq][irec] = buffer[4 * inv_acq->nrec + irec];
    inv_emf->d_Hx[1][ifreq][irec] = buffer[5 * inv_acq->nrec + irec];
    inv_emf->d_Hy[0][ifreq][irec] = buffer[6 * inv_acq->nrec + irec];
    inv_emf->d_Hy[1][ifreq][irec] = buffer[7 * inv_acq->nrec + irec];
  }
}

static void pack_adjoint_sources(int ifreq, float _Complex *buffer)
{
  int irec;
  for(irec = 0; irec < inv_acq->nrec; ++irec) {
    buffer[0 * inv_acq->nrec + irec] = inv_emf->s_Ex[0][ifreq][irec];
    buffer[1 * inv_acq->nrec + irec] = inv_emf->s_Ex[1][ifreq][irec];
    buffer[2 * inv_acq->nrec + irec] = inv_emf->s_Ey[0][ifreq][irec];
    buffer[3 * inv_acq->nrec + irec] = inv_emf->s_Ey[1][ifreq][irec];
    buffer[4 * inv_acq->nrec + irec] = inv_emf->s_Hx[0][ifreq][irec];
    buffer[5 * inv_acq->nrec + irec] = inv_emf->s_Hx[1][ifreq][irec];
    buffer[6 * inv_acq->nrec + irec] = inv_emf->s_Hy[0][ifreq][irec];
    buffer[7 * inv_acq->nrec + irec] = inv_emf->s_Hy[1][ifreq][irec];
  }
}

static void unpack_adjoint_sources(int ifreq, const float _Complex *buffer)
{
  int irec;
  for(irec = 0; irec < inv_acq->nrec; ++irec) {
    inv_emf->s_Ex[0][ifreq][irec] = buffer[0 * inv_acq->nrec + irec];
    inv_emf->s_Ex[1][ifreq][irec] = buffer[1 * inv_acq->nrec + irec];
    inv_emf->s_Ey[0][ifreq][irec] = buffer[2 * inv_acq->nrec + irec];
    inv_emf->s_Ey[1][ifreq][irec] = buffer[3 * inv_acq->nrec + irec];
    inv_emf->s_Hx[0][ifreq][irec] = buffer[4 * inv_acq->nrec + irec];
    inv_emf->s_Hx[1][ifreq][irec] = buffer[5 * inv_acq->nrec + irec];
    inv_emf->s_Hy[0][ifreq][irec] = buffer[6 * inv_acq->nrec + irec];
    inv_emf->s_Hy[1][ifreq][irec] = buffer[7 * inv_acq->nrec + irec];
  }
}

/* Worker-side solve for one frequency: run both source polarizations, cache the full
 * forward fields locally for the later adjoint gradient, and sample receiver responses
 * that rank 0 will convert into impedance residuals. */
static void solve_forward_frequency(int ifreq)
{
  int i, n, ipolar, lev;
  complex *u_bc;

  if(inv_emf->verb) printf("**** freq=%g\n", inv_emf->freqs[ifreq]);
  extend_model_init(inv_emf, ifreq);
  gmg_init(inv_emf, ifreq);
  n = 3 * (gmg[0].n1 + 1) * (gmg[0].n2 + 1) * (gmg[0].n3 + 1);
  u_bc = alloc1complex(n);

  for(ipolar = 0; ipolar < 2; ++ipolar) {
    if(inv_emf->verb) {
      if(ipolar == 0) printf("--- XY polarization ---\n");
      if(ipolar == 1) printf("--- YX polarization ---\n");
    }
    mt1d_efield_at_boundary(gmg, inv_emf->freqs[ifreq], ipolar);
    memcpy(u_bc, &gmg[0].u[0][0][0][0], n * sizeof(complex));
    memset(&gmg[0].f[0][0][0][0], 0, n * sizeof(complex));
    residual(gmg, 0);
    memcpy(&gmg[0].f[0][0][0][0], &gmg[0].r[0][0][0][0], n * sizeof(complex));
    memset(&gmg[0].r[0][0][0][0], 0, n * sizeof(complex));
    memset(&gmg[0].u[0][0][0][0], 0, n * sizeof(complex));
    for(icycle = 0; icycle < ncycle; ++icycle) {
      for(lev = 1; lev < lmax; ++lev) grid_init(gmg, lev);
      if(cycleopt == 1) v_cycle(gmg, 0);
      if(cycleopt == 2) f_cycle(gmg, 0);
      for(lev = 1; lev < lmax; ++lev) grid_free(gmg, lev);
    }
    for(i = 0; i < n; ++i) (&gmg[0].u[0][0][0][0])[i] += u_bc[i];
    compute_H_from_E(gmg);
    extract_mt_data(inv_acq, ifreq, ipolar, inv_emf->Efwd);
  }

  free1complex(u_bc);
  gmg_free();
  extend_model_free(inv_emf);
}

/* After rank 0 has converted receiver residuals into adjoint sources for this frequency,
 * the worker injects them, solves the adjoint systems, and collapses the local forward/adjoint
 * field products into this frequency's gradient contribution. */
static void solve_adjoint_frequency(int ifreq, float *g)
{
  int i, j, k;
  int id, n, ipolar, lev;
  int ncell = inv_emf->nx * inv_emf->ny * inv_emf->nz;

  if(inv_emf->verb) printf("**** freq=%g\n", inv_emf->freqs[ifreq]);
  extend_model_init(inv_emf, ifreq);
  gmg_init(inv_emf, ifreq);
  n = 3 * (gmg[0].n1 + 1) * (gmg[0].n2 + 1) * (gmg[0].n3 + 1);

  for(ipolar = 0; ipolar < 2; ++ipolar) {
    if(inv_emf->verb) {
      if(ipolar == 0) printf("--- XY polarization ---\n");
      if(ipolar == 1) printf("--- YX polarization ---\n");
    }
    inject_adjoint_sources(inv_acq, ifreq, ipolar);
    memset(&gmg[0].r[0][0][0][0], 0, n * sizeof(complex));
    memset(&gmg[0].u[0][0][0][0], 0, n * sizeof(complex));
    for(icycle = 0; icycle < ncycle; ++icycle) {
      for(lev = 1; lev < lmax; ++lev) grid_init(gmg, lev);
      if(cycleopt == 1) v_cycle(gmg, 0);
      if(cycleopt == 2) f_cycle(gmg, 0);
      for(lev = 1; lev < lmax; ++lev) grid_free(gmg, lev);
    }
    compute_H_from_E(gmg);
    extract_mt_data(inv_acq, ifreq, ipolar, inv_emf->Eadj);
  }

  gmg_free();
  extend_model_free(inv_emf);

  for(ipolar = 0; ipolar < 2; ++ipolar) {
    for(k = 0; k < inv_emf->nz; ++k) {
      for(j = 0; j < inv_emf->ny; ++j) {
        for(i = 0; i < inv_emf->nx; ++i) {
          id = i + inv_emf->nx * (j + inv_emf->ny * k);
          g[id] += crealf(inv_emf->Efwd[ipolar][ifreq][id] * inv_emf->Eadj[ipolar][ifreq][id]);
          g[id] += crealf(inv_emf->Efwd[ipolar][ifreq][id + ncell] * inv_emf->Eadj[ipolar][ifreq][id + ncell]);
          g[id + ncell] += crealf(inv_emf->Efwd[ipolar][ifreq][id + 2 * ncell] * inv_emf->Eadj[ipolar][ifreq][id + 2 * ncell]);
        }
      }
    }
  }
}

/* Rank 0 owns the objective evaluation bookkeeping: form impedances from the returned
 * receiver fields, accumulate the data misfit, and translate the impedance residuals into
 * adjoint source terms that are sent back to the worker handling the same frequency. */
static double compute_frequency_residual_and_sources(int ifreq)
{
  int irec;
  complex det, dZxxdu, dZxydu, dZyxdu, dZyydu;
  complex wcres_Zxx, wcres_Zxy, wcres_Zyx, wcres_Zyy;
  double fcost = 0.0;
  double det_abs;

  for(irec = 0; irec < inv_acq->nrec; ++irec) {
    float wxx2, wxy2, wyx2, wyy2;

    det = inv_emf->d_Hx[0][ifreq][irec] * inv_emf->d_Hy[1][ifreq][irec]
        - inv_emf->d_Hy[0][ifreq][irec] * inv_emf->d_Hx[1][ifreq][irec];
    det_abs = cabs(det);
    if(det_abs < 1e-30) err("singular MT impedance denominator at ifreq=%d irec=%d", ifreq, irec);

    inv_emf->cal_Zxx[ifreq][irec] = (inv_emf->d_Ex[0][ifreq][irec] * inv_emf->d_Hy[1][ifreq][irec]
                                   - inv_emf->d_Ex[1][ifreq][irec] * inv_emf->d_Hy[0][ifreq][irec]) / det;
    inv_emf->cal_Zxy[ifreq][irec] = (inv_emf->d_Ex[1][ifreq][irec] * inv_emf->d_Hx[0][ifreq][irec]
                                   - inv_emf->d_Ex[0][ifreq][irec] * inv_emf->d_Hx[1][ifreq][irec]) / det;
    inv_emf->cal_Zyx[ifreq][irec] = (inv_emf->d_Ey[0][ifreq][irec] * inv_emf->d_Hy[1][ifreq][irec]
                                   - inv_emf->d_Ey[1][ifreq][irec] * inv_emf->d_Hy[0][ifreq][irec]) / det;
    inv_emf->cal_Zyy[ifreq][irec] = (inv_emf->d_Ey[1][ifreq][irec] * inv_emf->d_Hx[0][ifreq][irec]
                                   - inv_emf->d_Ey[0][ifreq][irec] * inv_emf->d_Hx[1][ifreq][irec]) / det;

    inv_emf->res_Zxx[ifreq][irec] = inv_emf->cal_Zxx[ifreq][irec] - inv_emf->obs_Zxx[ifreq][irec];
    inv_emf->res_Zxy[ifreq][irec] = inv_emf->cal_Zxy[ifreq][irec] - inv_emf->obs_Zxy[ifreq][irec];
    inv_emf->res_Zyx[ifreq][irec] = inv_emf->cal_Zyx[ifreq][irec] - inv_emf->obs_Zyx[ifreq][irec];
    inv_emf->res_Zyy[ifreq][irec] = inv_emf->cal_Zyy[ifreq][irec] - inv_emf->obs_Zyy[ifreq][irec];

    wxx2 = inv_emf->w_Zxx[ifreq][irec] * inv_emf->w_Zxx[ifreq][irec];
    wxy2 = inv_emf->w_Zxy[ifreq][irec] * inv_emf->w_Zxy[ifreq][irec];
    wyx2 = inv_emf->w_Zyx[ifreq][irec] * inv_emf->w_Zyx[ifreq][irec];
    wyy2 = inv_emf->w_Zyy[ifreq][irec] * inv_emf->w_Zyy[ifreq][irec];

    wcres_Zxx = wxx2 * conj(inv_emf->res_Zxx[ifreq][irec]);
    wcres_Zxy = wxy2 * conj(inv_emf->res_Zxy[ifreq][irec]);
    wcres_Zyx = wyx2 * conj(inv_emf->res_Zyx[ifreq][irec]);
    wcres_Zyy = wyy2 * conj(inv_emf->res_Zyy[ifreq][irec]);
    fcost += 0.5 * (
        creal(wcres_Zxx * inv_emf->res_Zxx[ifreq][irec]) +
        creal(wcres_Zxy * inv_emf->res_Zxy[ifreq][irec]) +
        creal(wcres_Zyx * inv_emf->res_Zyx[ifreq][irec]) +
        creal(wcres_Zyy * inv_emf->res_Zyy[ifreq][irec]));

    dZxxdu =  inv_emf->d_Hy[1][ifreq][irec] / det;
    dZxydu = -inv_emf->d_Hx[1][ifreq][irec] / det;
    dZyxdu = 0.0;
    dZyydu = 0.0;
    inv_emf->s_Ex[0][ifreq][irec] = -(wcres_Zxx * dZxxdu + wcres_Zxy * dZxydu + wcres_Zyx * dZyxdu + wcres_Zyy * dZyydu);

    dZxxdu = 0.0;
    dZxydu = 0.0;
    dZyxdu =  inv_emf->d_Hy[1][ifreq][irec] / det;
    dZyydu = -inv_emf->d_Hx[1][ifreq][irec] / det;
    inv_emf->s_Ey[0][ifreq][irec] = -(wcres_Zxx * dZxxdu + wcres_Zxy * dZxydu + wcres_Zyx * dZyxdu + wcres_Zyy * dZyydu);

    dZxxdu = -inv_emf->cal_Zxx[ifreq][irec] * inv_emf->d_Hy[1][ifreq][irec] / det;
    dZxydu =  inv_emf->cal_Zxx[ifreq][irec] * inv_emf->d_Hx[1][ifreq][irec] / det;
    dZyxdu = -inv_emf->cal_Zyx[ifreq][irec] * inv_emf->d_Hy[1][ifreq][irec] / det;
    dZyydu =  inv_emf->cal_Zyx[ifreq][irec] * inv_emf->d_Hx[1][ifreq][irec] / det;
    inv_emf->s_Hx[0][ifreq][irec] = -(wcres_Zxx * dZxxdu + wcres_Zxy * dZxydu + wcres_Zyx * dZyxdu + wcres_Zyy * dZyydu);

    dZxxdu = -inv_emf->cal_Zxy[ifreq][irec] * inv_emf->d_Hy[1][ifreq][irec] / det;
    dZxydu =  inv_emf->cal_Zxy[ifreq][irec] * inv_emf->d_Hx[1][ifreq][irec] / det;
    dZyxdu = -inv_emf->cal_Zyy[ifreq][irec] * inv_emf->d_Hy[1][ifreq][irec] / det;
    dZyydu =  inv_emf->cal_Zyy[ifreq][irec] * inv_emf->d_Hx[1][ifreq][irec] / det;
    inv_emf->s_Hy[0][ifreq][irec] = -(wcres_Zxx * dZxxdu + wcres_Zxy * dZxydu + wcres_Zyx * dZyxdu + wcres_Zyy * dZyydu);

    dZxxdu = -inv_emf->d_Hy[0][ifreq][irec] / det;
    dZxydu =  inv_emf->d_Hx[0][ifreq][irec] / det;
    dZyxdu = 0.0;
    dZyydu = 0.0;
    inv_emf->s_Ex[1][ifreq][irec] = -(wcres_Zxx * dZxxdu + wcres_Zxy * dZxydu + wcres_Zyx * dZyxdu + wcres_Zyy * dZyydu);

    dZxxdu = 0.0;
    dZxydu = 0.0;
    dZyxdu = -inv_emf->d_Hy[0][ifreq][irec] / det;
    dZyydu =  inv_emf->d_Hx[0][ifreq][irec] / det;
    inv_emf->s_Ey[1][ifreq][irec] = -(wcres_Zxx * dZxxdu + wcres_Zxy * dZxydu + wcres_Zyx * dZyxdu + wcres_Zyy * dZyydu);

    dZxxdu =  inv_emf->cal_Zxx[ifreq][irec] * inv_emf->d_Hy[0][ifreq][irec] / det;
    dZxydu = -inv_emf->cal_Zxx[ifreq][irec] * inv_emf->d_Hx[0][ifreq][irec] / det;
    dZyxdu =  inv_emf->cal_Zyx[ifreq][irec] * inv_emf->d_Hy[0][ifreq][irec] / det;
    dZyydu = -inv_emf->cal_Zyx[ifreq][irec] * inv_emf->d_Hx[0][ifreq][irec] / det;
    inv_emf->s_Hx[1][ifreq][irec] = -(wcres_Zxx * dZxxdu + wcres_Zxy * dZxydu + wcres_Zyx * dZyxdu + wcres_Zyy * dZyydu);

    dZxxdu =  inv_emf->cal_Zxy[ifreq][irec] * inv_emf->d_Hy[0][ifreq][irec] / det;
    dZxydu = -inv_emf->cal_Zxy[ifreq][irec] * inv_emf->d_Hx[0][ifreq][irec] / det;
    dZyxdu =  inv_emf->cal_Zyy[ifreq][irec] * inv_emf->d_Hy[0][ifreq][irec] / det;
    dZyydu = -inv_emf->cal_Zyy[ifreq][irec] * inv_emf->d_Hx[0][ifreq][irec] / det;
    inv_emf->s_Hy[1][ifreq][irec] = -(wcres_Zxx * dZxxdu + wcres_Zxy * dZxydu + wcres_Zyx * dZyxdu + wcres_Zyy * dZyydu);
  }

  return fcost;
}

static void apply_log_parameter_chain_rule(const float *x, float *g)
{
  int i, j, k;
  int id;
  int ncell = inv_emf->nx * inv_emf->ny * inv_emf->nz;

  for(k = 0; k < inv_emf->nz; ++k) {
    for(j = 0; j < inv_emf->ny; ++j) {
      for(i = 0; i < inv_emf->nx; ++i) {
        float sigma_h, sigma_v;
        float rho_h, rho_v;

        id = i + inv_emf->nx * (j + inv_emf->ny * k);
        sigma_h = expf(x[id]);
        sigma_v = expf(x[id + ncell]);
        rho_h = expf(-x[id]);
        rho_v = expf(-x[id + ncell]);

        g[id] *= sigma_h;
        g[id + ncell] *= sigma_v;

        if(rho_h >= inv_emf->rho_air) g[id] = 0.0f;
        if(rho_v >= inv_emf->rho_air) g[id + ncell] = 0.0f;
      }
    }
  }
}

/* Allocate only the buffers needed by each rank. Rank 0 keeps the optimizer-facing
 * tensors (observed data, calculated impedances, residuals, adjoint sources), while
 * workers allocate the volumetric forward/adjoint field caches required for one-frequency
 * solves. This avoids carrying the full field history on the master process. */
void inversion_init(acq_t *acq, emf_t *emf)
{
  int rank, size;
  int ncell;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  inv_acq = acq;
  inv_emf = emf;
  clear_inversion_pointers(emf);

  emf->d_Ex = alloc3complexf(acq->nrec, emf->nfreq, 2);
  emf->d_Ey = alloc3complexf(acq->nrec, emf->nfreq, 2);
  emf->d_Hx = alloc3complexf(acq->nrec, emf->nfreq, 2);
  emf->d_Hy = alloc3complexf(acq->nrec, emf->nfreq, 2);
  memset(&emf->d_Ex[0][0][0], 0, 2 * acq->nrec * emf->nfreq * sizeof(float _Complex));
  memset(&emf->d_Ey[0][0][0], 0, 2 * acq->nrec * emf->nfreq * sizeof(float _Complex));
  memset(&emf->d_Hx[0][0][0], 0, 2 * acq->nrec * emf->nfreq * sizeof(float _Complex));
  memset(&emf->d_Hy[0][0][0], 0, 2 * acq->nrec * emf->nfreq * sizeof(float _Complex));

  emf->s_Ex = alloc3complexf(acq->nrec, emf->nfreq, 2);
  emf->s_Ey = alloc3complexf(acq->nrec, emf->nfreq, 2);
  emf->s_Hx = alloc3complexf(acq->nrec, emf->nfreq, 2);
  emf->s_Hy = alloc3complexf(acq->nrec, emf->nfreq, 2);
  memset(&emf->s_Ex[0][0][0], 0, 2 * acq->nrec * emf->nfreq * sizeof(float _Complex));
  memset(&emf->s_Ey[0][0][0], 0, 2 * acq->nrec * emf->nfreq * sizeof(float _Complex));
  memset(&emf->s_Hx[0][0][0], 0, 2 * acq->nrec * emf->nfreq * sizeof(float _Complex));
  memset(&emf->s_Hy[0][0][0], 0, 2 * acq->nrec * emf->nfreq * sizeof(float _Complex));

  ncell = emf->nx * emf->ny * emf->nz;
  if(size == 1 || rank != 0) {
    emf->Efwd = alloc3complexf(3 * ncell, emf->nfreq, 2);
    emf->Eadj = alloc3complexf(3 * ncell, emf->nfreq, 2);
    memset(&emf->Efwd[0][0][0], 0, 2 * emf->nfreq * 3 * ncell * sizeof(float _Complex));
    memset(&emf->Eadj[0][0][0], 0, 2 * emf->nfreq * 3 * ncell * sizeof(float _Complex));
  }

  if(rank == 0) {
    emf->cal_Zxx = alloc2complexf(acq->nrec, emf->nfreq);
    emf->cal_Zxy = alloc2complexf(acq->nrec, emf->nfreq);
    emf->cal_Zyx = alloc2complexf(acq->nrec, emf->nfreq);
    emf->cal_Zyy = alloc2complexf(acq->nrec, emf->nfreq);
    emf->obs_Zxx = alloc2complexf(acq->nrec, emf->nfreq);
    emf->obs_Zxy = alloc2complexf(acq->nrec, emf->nfreq);
    emf->obs_Zyx = alloc2complexf(acq->nrec, emf->nfreq);
    emf->obs_Zyy = alloc2complexf(acq->nrec, emf->nfreq);
    emf->res_Zxx = alloc2complexf(acq->nrec, emf->nfreq);
    emf->res_Zxy = alloc2complexf(acq->nrec, emf->nfreq);
    emf->res_Zyx = alloc2complexf(acq->nrec, emf->nfreq);
    emf->res_Zyy = alloc2complexf(acq->nrec, emf->nfreq);
    emf->w_Zxx = alloc2float(acq->nrec, emf->nfreq);
    emf->w_Zxy = alloc2float(acq->nrec, emf->nfreq);
    emf->w_Zyx = alloc2float(acq->nrec, emf->nfreq);
    emf->w_Zyy = alloc2float(acq->nrec, emf->nfreq);
    memset(&emf->cal_Zxx[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float _Complex));
    memset(&emf->cal_Zxy[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float _Complex));
    memset(&emf->cal_Zyx[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float _Complex));
    memset(&emf->cal_Zyy[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float _Complex));
    memset(&emf->res_Zxx[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float _Complex));
    memset(&emf->res_Zxy[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float _Complex));
    memset(&emf->res_Zyx[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float _Complex));
    memset(&emf->res_Zyy[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float _Complex));
    memset(&emf->w_Zxx[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float));
    memset(&emf->w_Zxy[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float));
    memset(&emf->w_Zyx[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float));
    memset(&emf->w_Zyy[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float));
  }
}

void inversion_free(emf_t *emf)
{
  if(emf->d_Ex != NULL) free3complexf(emf->d_Ex);
  if(emf->d_Ey != NULL) free3complexf(emf->d_Ey);
  if(emf->d_Hx != NULL) free3complexf(emf->d_Hx);
  if(emf->d_Hy != NULL) free3complexf(emf->d_Hy);
  if(emf->cal_Zxx != NULL) free2complexf(emf->cal_Zxx);
  if(emf->cal_Zxy != NULL) free2complexf(emf->cal_Zxy);
  if(emf->cal_Zyx != NULL) free2complexf(emf->cal_Zyx);
  if(emf->cal_Zyy != NULL) free2complexf(emf->cal_Zyy);
  if(emf->obs_Zxx != NULL) free2complexf(emf->obs_Zxx);
  if(emf->obs_Zxy != NULL) free2complexf(emf->obs_Zxy);
  if(emf->obs_Zyx != NULL) free2complexf(emf->obs_Zyx);
  if(emf->obs_Zyy != NULL) free2complexf(emf->obs_Zyy);
  if(emf->res_Zxx != NULL) free2complexf(emf->res_Zxx);
  if(emf->res_Zxy != NULL) free2complexf(emf->res_Zxy);
  if(emf->res_Zyx != NULL) free2complexf(emf->res_Zyx);
  if(emf->res_Zyy != NULL) free2complexf(emf->res_Zyy);
  if(emf->w_Zxx != NULL) free2float(emf->w_Zxx);
  if(emf->w_Zxy != NULL) free2float(emf->w_Zxy);
  if(emf->w_Zyx != NULL) free2float(emf->w_Zyx);
  if(emf->w_Zyy != NULL) free2float(emf->w_Zyy);
  if(emf->s_Ex != NULL) free3complexf(emf->s_Ex);
  if(emf->s_Ey != NULL) free3complexf(emf->s_Ey);
  if(emf->s_Hx != NULL) free3complexf(emf->s_Hx);
  if(emf->s_Hy != NULL) free3complexf(emf->s_Hy);
  if(emf->Efwd != NULL) free3complexf(emf->Efwd);
  if(emf->Eadj != NULL) free3complexf(emf->Eadj);

  clear_inversion_pointers(emf);
}

/* Long-lived worker loop for inversion. Each optimization step starts with a broadcast
 * of the current model vector, then workers repeatedly: (1) solve one assigned frequency
 * forward, (2) return sampled receiver fields, (3) receive adjoint sources for that same
 * frequency, and (4) return the per-frequency gradient contribution. */
void inversion_worker_loop(acq_t *acq, emf_t *emf)
{
  int command;
  int ncell = emf->nx * emf->ny * emf->nz;
  float *x = alloc1float(2 * ncell);
  float *g_local = alloc1float(2 * ncell);
  float _Complex *forward_data = alloc1complexf(8 * acq->nrec);
  float _Complex *source_data = alloc1complexf(8 * acq->nrec);

  inv_acq = acq;
  inv_emf = emf;

  while(1) {
    int assigned_ifreq;
    MPI_Status status;

    MPI_Bcast(&command, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(command == INV_CMD_STOP) break;
    if(command != INV_CMD_EVAL) err("unknown inversion MPI command %d", command);

    MPI_Bcast(x, 2 * ncell, MPI_FLOAT, 0, MPI_COMM_WORLD);
    model_from_vector(x);

    while(1) {
      MPI_Recv(&assigned_ifreq, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      if(status.MPI_TAG == TAG_INV_PHASE_STOP) break;

      solve_forward_frequency(assigned_ifreq);
      pack_forward_data(assigned_ifreq, forward_data);
      MPI_Send(&assigned_ifreq, 1, MPI_INT, 0, TAG_INV_FORWARD_INDEX, MPI_COMM_WORLD);
      MPI_Send(forward_data, 8 * acq->nrec * (int)sizeof(float _Complex), MPI_BYTE, 0, TAG_INV_FORWARD_DATA, MPI_COMM_WORLD);

      MPI_Recv(source_data, 8 * acq->nrec * (int)sizeof(float _Complex), MPI_BYTE, 0, TAG_INV_ADJOINT_SOURCES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      unpack_adjoint_sources(assigned_ifreq, source_data);

      memset(g_local, 0, 2 * ncell * sizeof(float));
      solve_adjoint_frequency(assigned_ifreq, g_local);
      MPI_Send(&assigned_ifreq, 1, MPI_INT, 0, TAG_INV_GRAD_INDEX, MPI_COMM_WORLD);
      MPI_Send(g_local, 2 * ncell, MPI_FLOAT, 0, TAG_INV_GRAD_DATA, MPI_COMM_WORLD);
    }
  }

  free1float(x);
  free1float(g_local);
  free1complexf(forward_data);
  free1complexf(source_data);
}

/* Objective callback executed only on rank 0. In serial it follows the original full
 * forward-then-adjoint path. Under MPI it becomes a scheduler: broadcast the current model,
 * dynamically assign frequencies to workers, assemble residuals and adjoint sources as soon
 * as each forward result arrives, then collect and sum the returned per-frequency gradients. */
float inversion_grad(const float *x, float *g)
{
  int rank, size;
  int ifreq;
  int ncell = inv_emf->nx * inv_emf->ny * inv_emf->nz;
  double fcost = 0.0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(rank != 0) err("inversion_grad must only run on MPI rank 0");

  memset(g, 0, 2 * ncell * sizeof(float));
  model_from_vector(x);

  if(size == 1) {
    zero_master_buffers();
    memset(&inv_emf->Efwd[0][0][0], 0, 2 * inv_emf->nfreq * 3 * ncell * sizeof(float _Complex));
    memset(&inv_emf->Eadj[0][0][0], 0, 2 * inv_emf->nfreq * 3 * ncell * sizeof(float _Complex));

    if(inv_emf->verb) printf("---------- forward modelling -----------\n");
    for(ifreq = 0; ifreq < inv_emf->nfreq; ++ifreq) solve_forward_frequency(ifreq);

    if(inv_emf->verb) printf("-------- compute adjoint sources ---------\n");
    for(ifreq = 0; ifreq < inv_emf->nfreq; ++ifreq) fcost += compute_frequency_residual_and_sources(ifreq);
    if(inv_emf->verb) printf("fcost=%g\n", fcost);
    
    if(inv_emf->verb) printf("---------- adjoint modelling -----------\n");
    for(ifreq = 0; ifreq < inv_emf->nfreq; ++ifreq) solve_adjoint_frequency(ifreq, g);

    apply_log_parameter_chain_rule(x, g);
    return (float)fcost;
  }

  {
    int command = INV_CMD_EVAL;
    int next_task = 0;
    int active_workers = 0;
    int worker;
    float *g_local = alloc1float(2 * ncell);
    float _Complex *forward_data = alloc1complexf(8 * inv_acq->nrec);
    float _Complex *source_data = alloc1complexf(8 * inv_acq->nrec);

    zero_master_buffers();

    MPI_Bcast(&command, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)x, 2 * ncell, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(inv_emf->verb) printf("---------- forward/adjoint modelling -----------\n");

    for(worker = 1; worker < size; ++worker) {
      if(next_task < inv_emf->nfreq) {
        MPI_Send(&next_task, 1, MPI_INT, worker, TAG_INV_WORK, MPI_COMM_WORLD);
        ++next_task;
        ++active_workers;
      } else {
        MPI_Send(NULL, 0, MPI_INT, worker, TAG_INV_PHASE_STOP, MPI_COMM_WORLD);
      }
    }

    while(active_workers > 0) {
      int result_ifreq;
      int grad_ifreq;
      MPI_Status status;

      MPI_Recv(&result_ifreq, 1, MPI_INT, MPI_ANY_SOURCE, TAG_INV_FORWARD_INDEX, MPI_COMM_WORLD, &status);
      worker = status.MPI_SOURCE;
      MPI_Recv(forward_data, 8 * inv_acq->nrec * (int)sizeof(float _Complex), MPI_BYTE, worker, TAG_INV_FORWARD_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      unpack_forward_data(result_ifreq, forward_data);

      fcost += compute_frequency_residual_and_sources(result_ifreq);
      pack_adjoint_sources(result_ifreq, source_data);
      MPI_Send(source_data, 8 * inv_acq->nrec * (int)sizeof(float _Complex), MPI_BYTE, worker, TAG_INV_ADJOINT_SOURCES, MPI_COMM_WORLD);

      MPI_Recv(&grad_ifreq, 1, MPI_INT, worker, TAG_INV_GRAD_INDEX, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(g_local, 2 * ncell, MPI_FLOAT, worker, TAG_INV_GRAD_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if(grad_ifreq != result_ifreq) err("worker %d returned mismatched gradient frequency %d for forward frequency %d", worker, grad_ifreq, result_ifreq);
      for(ifreq = 0; ifreq < 2 * ncell; ++ifreq) g[ifreq] += g_local[ifreq];

      if(inv_emf->verb) printf("rank 0 collected inversion freq=%g from worker %d\n", inv_emf->freqs[result_ifreq], worker);

      if(next_task < inv_emf->nfreq) {
        MPI_Send(&next_task, 1, MPI_INT, worker, TAG_INV_WORK, MPI_COMM_WORLD);
        ++next_task;
      } else {
        MPI_Send(NULL, 0, MPI_INT, worker, TAG_INV_PHASE_STOP, MPI_COMM_WORLD);
        --active_workers;
      }
    }

    apply_log_parameter_chain_rule(x, g);
    free1float(g_local);
    free1complexf(forward_data);
    free1complexf(source_data);
  }

  return (float)fcost;
}
