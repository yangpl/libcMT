/* Forward-modelling driver for MT responses.
 * Runs the multigrid solve for each frequency and polarization, extracts
 * receiver fields, converts them to impedance tensors, and writes output.
 *----------------------------------------------------------------------
 *   Copyright (c) Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *----------------------------------------------------------------------
 */
#include "cstd.h"
#include "acq.h"
#include "gmg.h"
#include <mpi.h>

#define TAG_WORK 1
#define TAG_RESULT_INDEX 2
#define TAG_RESULT_DATA 3
#define TAG_STOP 4

void extend_model_init(emf_t *emf, int ifreq);
void extend_model_free(emf_t *emf);

void mt1d_efield_at_boundary(gmg_t *gmg, double freq, int ipolar);
void extract_mt_data(acq_t *acq, int ifreq, int ipolar, float _Complex ***E);
void write_mt_data(acq_t *acq, emf_t *emf, char *fname);

static void allocate_modelling_buffers(acq_t *acq, emf_t *emf, int need_fields)
{
  emf->d_Ex = NULL;
  emf->d_Ey = NULL;
  emf->d_Hx = NULL;
  emf->d_Hy = NULL;
  if (need_fields) {
    emf->d_Ex = alloc3complexf(acq->nrec, emf->nfreq, 2);
    emf->d_Ey = alloc3complexf(acq->nrec, emf->nfreq, 2);
    emf->d_Hx = alloc3complexf(acq->nrec, emf->nfreq, 2);
    emf->d_Hy = alloc3complexf(acq->nrec, emf->nfreq, 2);
    memset(&emf->d_Ex[0][0][0], 0, 2 * acq->nrec * emf->nfreq * sizeof(float _Complex));
    memset(&emf->d_Ey[0][0][0], 0, 2 * acq->nrec * emf->nfreq * sizeof(float _Complex));
    memset(&emf->d_Hx[0][0][0], 0, 2 * acq->nrec * emf->nfreq * sizeof(float _Complex));
    memset(&emf->d_Hy[0][0][0], 0, 2 * acq->nrec * emf->nfreq * sizeof(float _Complex));
  }

  emf->cal_Zxx = alloc2complexf(acq->nrec, emf->nfreq);
  emf->cal_Zxy = alloc2complexf(acq->nrec, emf->nfreq);
  emf->cal_Zyx = alloc2complexf(acq->nrec, emf->nfreq);
  emf->cal_Zyy = alloc2complexf(acq->nrec, emf->nfreq);
  memset(&emf->cal_Zxx[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float _Complex));
  memset(&emf->cal_Zxy[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float _Complex));
  memset(&emf->cal_Zyx[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float _Complex));
  memset(&emf->cal_Zyy[0][0], 0, (size_t)emf->nfreq * acq->nrec * sizeof(float _Complex));
}

static void free_modelling_buffers(emf_t *emf)
{
  if (emf->d_Ex != NULL) free3complexf(emf->d_Ex);
  if (emf->d_Ey != NULL) free3complexf(emf->d_Ey);
  if (emf->d_Hx != NULL) free3complexf(emf->d_Hx);
  if (emf->d_Hy != NULL) free3complexf(emf->d_Hy);
  free2complexf(emf->cal_Zxx);
  free2complexf(emf->cal_Zxy);
  free2complexf(emf->cal_Zyx);
  free2complexf(emf->cal_Zyy);
}

static void solve_frequency(acq_t *acq, emf_t *emf, int ifreq, float _Complex *result)
{
  int i, irec, n, ipolar, lev;
  complex det, *u_bc;
  double det_abs;

  if (emf->verb) printf("**** freq=%g\n", emf->freqs[ifreq]);
  extend_model_init(emf, ifreq);
  gmg_init(emf, ifreq);
  n = 3 * (gmg[0].n1 + 1) * (gmg[0].n2 + 1) * (gmg[0].n3 + 1);
  u_bc = alloc1complex(n);

  /* Run the two source polarizations needed to assemble the 2x2 impedance tensor. */
  for (ipolar = 0; ipolar < 2; ++ipolar) {
    if (emf->verb) {
      if (ipolar == 0) printf("--- XY polarization ---\n");
      if (ipolar == 1) printf("--- YX polarization ---\n");
    }
    /* Seed the 3D solve with the 1D boundary field for this polarization. */
    mt1d_efield_at_boundary(gmg, emf->freqs[ifreq], ipolar);
    memcpy(u_bc, &gmg[0].u[0][0][0][0], n * sizeof(complex));
    memset(&gmg[0].f[0][0][0][0], 0, n * sizeof(complex));
    residual(gmg, 0);//r=-A*u_bc
    memcpy(&gmg[0].f[0][0][0][0], &gmg[0].r[0][0][0][0], n * sizeof(complex));
    memset(&gmg[0].r[0][0][0][0], 0, n * sizeof(complex));
    memset(&gmg[0].u[0][0][0][0], 0, n * sizeof(complex));
    for (icycle = 0; icycle < ncycle; ++icycle) {
      for (lev = 1; lev < lmax; ++lev) grid_init(gmg, lev);
      if (cycleopt == 1) v_cycle(gmg, 0);
      if (cycleopt == 2) f_cycle(gmg, 0);
      for (lev = 1; lev < lmax; ++lev) grid_free(gmg, lev);
    }
    /* Add the boundary field back to the solved perturbation to recover total E. */
    for (i = 0; i < n; ++i) (&gmg[0].u[0][0][0][0])[i] += u_bc[i];//then E=gmg[0].u
    compute_H_from_E(gmg);//then H=gmg[0].f
    extract_mt_data(acq, ifreq, ipolar, NULL);//set E=NULL to avoid storing E fields
  }
  free1complex(u_bc);

  /* Convert the sampled E/H fields into impedance tensor entries at each receiver. */
  for (irec = 0; irec < acq->nrec; ++irec) {
    det = emf->d_Hx[0][ifreq][irec] * emf->d_Hy[1][ifreq][irec] - emf->d_Hy[0][ifreq][irec] * emf->d_Hx[1][ifreq][irec];
    det_abs = cabs(det);
    if(det_abs < 1e-30) err("singular MT impedance denominator at ifreq=%d irec=%d", ifreq, irec);
    //convert E/H to impedance Z
    emf->cal_Zxx[ifreq][irec] = (emf->d_Ex[0][ifreq][irec] * emf->d_Hy[1][ifreq][irec] - emf->d_Ex[1][ifreq][irec] * emf->d_Hy[0][ifreq][irec]) / det;
    emf->cal_Zxy[ifreq][irec] = (emf->d_Ex[1][ifreq][irec] * emf->d_Hx[0][ifreq][irec] - emf->d_Ex[0][ifreq][irec] * emf->d_Hx[1][ifreq][irec]) / det;
    emf->cal_Zyx[ifreq][irec] = (emf->d_Ey[0][ifreq][irec] * emf->d_Hy[1][ifreq][irec] - emf->d_Ey[1][ifreq][irec] * emf->d_Hy[0][ifreq][irec]) / det;
    emf->cal_Zyy[ifreq][irec] = (emf->d_Ey[1][ifreq][irec] * emf->d_Hx[0][ifreq][irec] - emf->d_Ey[0][ifreq][irec] * emf->d_Hx[1][ifreq][irec]) / det;
  }

  if (result != NULL) {
    for (irec = 0; irec < acq->nrec; ++irec) {
      result[0 * acq->nrec + irec] = emf->cal_Zxx[ifreq][irec];
      result[1 * acq->nrec + irec] = emf->cal_Zxy[ifreq][irec];
      result[2 * acq->nrec + irec] = emf->cal_Zyx[ifreq][irec];
      result[3 * acq->nrec + irec] = emf->cal_Zyy[ifreq][irec];
    }
  }

  gmg_free();
  extend_model_free(emf);
}

static void store_frequency_result(acq_t *acq, emf_t *emf, int ifreq, const float _Complex *result)
{
  int irec;

  for (irec = 0; irec < acq->nrec; ++irec) {
    emf->cal_Zxx[ifreq][irec] = result[0 * acq->nrec + irec];
    emf->cal_Zxy[ifreq][irec] = result[1 * acq->nrec + irec];
    emf->cal_Zyx[ifreq][irec] = result[2 * acq->nrec + irec];
    emf->cal_Zyy[ifreq][irec] = result[3 * acq->nrec + irec];
  }
}

/* Solve the forward MT problem for every frequency and polarization, then write the responses. */
void do_modelling(acq_t *acq, emf_t *emf)
{
  int ifreq;
  int rank, size;
  char *fdata;
  
  if(!getparstring("fdata", &fdata)) fdata="mt_data.h5";

  /* Parallelization is across frequency, using MPI ranks instead of threads.
   * Each rank runs an independent multigrid solve for one assigned frequency,
   * so workers never touch each other's GMG state or field buffers.
   */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Serial fallback: with only one rank, compute all frequencies locally and
   * keep the full E/H field buffers because the same process will also form Z.
   */
  if (size == 1) {
    allocate_modelling_buffers(acq, emf, 1);
    for (ifreq = 0; ifreq < emf->nfreq; ++ifreq) solve_frequency(acq, emf, ifreq, NULL);
    write_mt_data(acq, emf, fdata);
    free_modelling_buffers(emf);
    return;
  }

  /* Rank 0 acts as the scheduler and collector.
   * It does not solve frequencies itself in the MPI case. Instead it:
   * 1) sends frequency indices to workers,
   * 2) receives one finished impedance tensor at a time,
   * 3) stores the result in the global output arrays,
   * 4) sends the worker another frequency or a stop message.
   *
   * Because only rank 0 writes the final cal_Z** arrays that are later written
   * to disk, output assembly is centralized and deterministic.
   */
  if (rank == 0) {
    int next_task, active_workers, worker;
    int result_ifreq;
    MPI_Status status;
    float _Complex *result;

    /* The master only needs storage for the final impedance tensors.
     * It does not compute fields, so it skips allocation of d_Ex/d_Ey/d_Hx/d_Hy.
     */
    allocate_modelling_buffers(acq, emf, 0);
    result = alloc1complexf(4 * acq->nrec);
    next_task = 0;
    active_workers = 0;

    /* Initial dispatch: give each worker at most one frequency.
     * If there are more workers than frequencies, the extra workers are stopped
     * immediately so they do not sit in the receive loop indefinitely.
     */
    for (worker = 1; worker < size; ++worker) {
      if (next_task < emf->nfreq) {
        MPI_Send(&next_task, 1, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
        ++next_task;
        ++active_workers;
      } else {
        MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
      }
    }

    /* Dynamic scheduling loop.
     * MPI_ANY_SOURCE lets rank 0 accept whichever worker finishes first.
     * This avoids load imbalance when some frequencies take longer to solve.
     */
    while (active_workers > 0) {
      /* First receive the frequency index, then receive the packed tensor data
       * for that frequency from the same worker.
       */
      MPI_Recv(&result_ifreq, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT_INDEX, MPI_COMM_WORLD, &status);
      worker = status.MPI_SOURCE;
      MPI_Recv(result, 4 * acq->nrec * (int)sizeof(float _Complex), MPI_BYTE, worker, TAG_RESULT_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      store_frequency_result(acq, emf, result_ifreq, result);

      if (emf->verb) printf("rank 0 collected freq=%g from worker %d\n", emf->freqs[result_ifreq], worker);

      /* Reuse the same worker immediately if there is more work.
       * Otherwise send TAG_STOP and mark that worker as inactive.
       */
      if (next_task < emf->nfreq) {
        MPI_Send(&next_task, 1, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
        ++next_task;
      } else {
        MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
        --active_workers;
      }
    }

    free1complexf(result);
    write_mt_data(acq, emf, fdata);
    free_modelling_buffers(emf);
    return;
  }

  /* Only worker ranks reach this block; rank 0 returns above. */
  {
    int assigned_ifreq;
    float _Complex *result;
    MPI_Status status;

    /* Worker ranks execute this block.
     * Each worker owns a private copy of emf/gmg state in its own MPI process,
     * so the multigrid solve is embarrassingly parallel across frequencies.
     * There is no shared-memory synchronization here; coordination happens only
     * through MPI messages with rank 0.
     */
    allocate_modelling_buffers(acq, emf, 1);
    result = alloc1complexf(4 * acq->nrec);
    while (1) {
      /* Wait for either:
       * TAG_WORK: solve the assigned frequency and send back 4 impedance arrays
       * TAG_STOP: no more work remains, so exit cleanly
       */
      MPI_Recv(&assigned_ifreq, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      /* TAG_STOP exits the worker loop immediately, so the solve/send path below only runs for real work messages. */
      if (status.MPI_TAG == TAG_STOP) break;

      /* solve_frequency() fills the worker-local field buffers, forms the
       * impedance tensor for one frequency, and packs Zxx/Zxy/Zyx/Zyy into
       * result[] for transmission back to rank 0.
       */
      solve_frequency(acq, emf, assigned_ifreq, result);
      MPI_Send(&assigned_ifreq, 1, MPI_INT, 0, TAG_RESULT_INDEX, MPI_COMM_WORLD);
      MPI_Send(result, 4 * acq->nrec * (int)sizeof(float _Complex), MPI_BYTE, 0, TAG_RESULT_DATA, MPI_COMM_WORLD);
    }
    free1complexf(result);
    free_modelling_buffers(emf);
  }
}
