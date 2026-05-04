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
void extract_mt_data(acq_t *acq, int ifreq, int ipolar);
void write_mt_data(acq_t *acq, emf_t *emf, char *fname);
void write_computational_model_hdf5(emf_t *emf, int ifreq, const char *fname);

static void make_frequency_mesh_filename(const char *base, int nfreq, int ifreq,
                                         char *fname, size_t fname_size)
{
  char *dot;
  size_t stem_len;

  if(nfreq == 1) {
    snprintf(fname, fname_size, "%s", base);
    return;
  }

  dot = strrchr(base, '.');
  if(dot != NULL) {
    stem_len = (size_t)(dot - base);
    snprintf(fname, fname_size, "%.*s_freq%04d%s", (int)stem_len, base, ifreq, dot);
  } else {
    snprintf(fname, fname_size, "%s_freq%04d.h5", base, ifreq);
  }
}

static void allocate_modelling_buffers(acq_t *acq, emf_t *emf, int nfreq_slots, int need_fields)
{
  emf->d_Ex = NULL;
  emf->d_Ey = NULL;
  emf->d_Hx = NULL;
  emf->d_Hy = NULL;
  if (need_fields) {
    emf->d_Ex = alloc3complexf(acq->nrec, nfreq_slots, 2);
    emf->d_Ey = alloc3complexf(acq->nrec, nfreq_slots, 2);
    emf->d_Hx = alloc3complexf(acq->nrec, nfreq_slots, 2);
    emf->d_Hy = alloc3complexf(acq->nrec, nfreq_slots, 2);
    if (emf->d_Ex == NULL || emf->d_Ey == NULL ||
        emf->d_Hx == NULL || emf->d_Hy == NULL)
      err("unable to allocate receiver field buffers");
    memset(&emf->d_Ex[0][0][0], 0, 2 * acq->nrec * nfreq_slots * sizeof(float _Complex));
    memset(&emf->d_Ey[0][0][0], 0, 2 * acq->nrec * nfreq_slots * sizeof(float _Complex));
    memset(&emf->d_Hx[0][0][0], 0, 2 * acq->nrec * nfreq_slots * sizeof(float _Complex));
    memset(&emf->d_Hy[0][0][0], 0, 2 * acq->nrec * nfreq_slots * sizeof(float _Complex));
  }

  emf->cal_Zxx = alloc2complexf(acq->nrec, nfreq_slots);
  emf->cal_Zxy = alloc2complexf(acq->nrec, nfreq_slots);
  emf->cal_Zyx = alloc2complexf(acq->nrec, nfreq_slots);
  emf->cal_Zyy = alloc2complexf(acq->nrec, nfreq_slots);
  if (emf->cal_Zxx == NULL || emf->cal_Zxy == NULL ||
      emf->cal_Zyx == NULL || emf->cal_Zyy == NULL)
    err("unable to allocate MT impedance buffers");
  memset(&emf->cal_Zxx[0][0], 0, (size_t)nfreq_slots * acq->nrec * sizeof(float _Complex));
  memset(&emf->cal_Zxy[0][0], 0, (size_t)nfreq_slots * acq->nrec * sizeof(float _Complex));
  memset(&emf->cal_Zyx[0][0], 0, (size_t)nfreq_slots * acq->nrec * sizeof(float _Complex));
  memset(&emf->cal_Zyy[0][0], 0, (size_t)nfreq_slots * acq->nrec * sizeof(float _Complex));
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

static void solve_frequency(acq_t *acq, emf_t *emf, int ifreq, int data_ifreq, float _Complex *result)
{
  int i, irec, n, ipolar, lev;
  complex det, *u_bc;
  double det_abs;

  if (emf->verb) printf("freq=%g\n", emf->freqs[ifreq]);
  extend_model_init(emf, ifreq);
  {
    char *fmesh;
    char mesh_fname[PATH_MAX];

    if(!getparstring("fmesh", &fmesh)) fmesh = "mesh.h5";
    make_frequency_mesh_filename(fmesh, emf->nfreq, ifreq, mesh_fname, sizeof(mesh_fname));
    write_computational_model_hdf5(emf, ifreq, mesh_fname);
    if(emf->verb) printf("wrote computational model mesh to %s\n", mesh_fname);
  }
  gmg_init(emf, ifreq);
  n = 3 * (gmg[0].n1 + 1) * (gmg[0].n2 + 1) * (gmg[0].n3 + 1);
  u_bc = alloc1complex(n);
  if (u_bc == NULL) err("unable to allocate boundary-field buffer");

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
    extract_mt_data(acq, data_ifreq, ipolar);
  }
  free1complex(u_bc);

  /* Convert the sampled E/H fields into impedance tensor entries at each receiver. */
  for (irec = 0; irec < acq->nrec; ++irec) {
    det = emf->d_Hx[0][data_ifreq][irec] * emf->d_Hy[1][data_ifreq][irec] - emf->d_Hy[0][data_ifreq][irec] * emf->d_Hx[1][data_ifreq][irec];
    det_abs = cabs(det);
    if(det_abs < 1e-30) err("singular MT impedance denominator at ifreq=%d irec=%d", ifreq, irec);
    //convert E/H to impedance Z
    emf->cal_Zxx[data_ifreq][irec] = (emf->d_Ex[0][data_ifreq][irec] * emf->d_Hy[1][data_ifreq][irec] - emf->d_Ex[1][data_ifreq][irec] * emf->d_Hy[0][data_ifreq][irec]) / det;
    emf->cal_Zxy[data_ifreq][irec] = (emf->d_Ex[1][data_ifreq][irec] * emf->d_Hx[0][data_ifreq][irec] - emf->d_Ex[0][data_ifreq][irec] * emf->d_Hx[1][data_ifreq][irec]) / det;
    emf->cal_Zyx[data_ifreq][irec] = (emf->d_Ey[0][data_ifreq][irec] * emf->d_Hy[1][data_ifreq][irec] - emf->d_Ey[1][data_ifreq][irec] * emf->d_Hy[0][data_ifreq][irec]) / det;
    emf->cal_Zyy[data_ifreq][irec] = (emf->d_Ey[1][data_ifreq][irec] * emf->d_Hx[0][data_ifreq][irec] - emf->d_Ey[0][data_ifreq][irec] * emf->d_Hx[1][data_ifreq][irec]) / det;
  }

  if (result != NULL) {
    for (irec = 0; irec < acq->nrec; ++irec) {
      result[0 * acq->nrec + irec] = emf->cal_Zxx[data_ifreq][irec];
      result[1 * acq->nrec + irec] = emf->cal_Zxy[data_ifreq][irec];
      result[2 * acq->nrec + irec] = emf->cal_Zyx[data_ifreq][irec];
      result[3 * acq->nrec + irec] = emf->cal_Zyy[data_ifreq][irec];
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
    allocate_modelling_buffers(acq, emf, emf->nfreq, 1);
    for (ifreq = 0; ifreq < emf->nfreq; ++ifreq) solve_frequency(acq, emf, ifreq, ifreq, NULL);
    write_mt_data(acq, emf, fdata);
    free_modelling_buffers(emf);
    return;
  }

  /* Rank 0 acts as scheduler and collector, but also solves reserved local
   * frequencies. This follows the inversion scheduler pattern: workers receive
   * dynamically assigned frequencies, while rank 0 services completed worker
   * results whenever they are ready and otherwise advances the remaining queue
   * by doing local modelling work.
   *
   * Because only rank 0 writes the final cal_Z** arrays that are later written
   * to disk, output assembly is centralized and deterministic.
   */
  if (rank == 0) {
    int next_task, active_workers, worker;
    int initial_worker_limit;
    int result_ifreq;
    MPI_Status status;
    float _Complex *result;
    int flag;

    /* Rank 0 keeps full final impedance arrays for output and full receiver
     * field arrays for its local solves. Worker ranks below allocate one slot.
     */
    allocate_modelling_buffers(acq, emf, emf->nfreq, 1);
    result = alloc1complexf(4 * acq->nrec);
    if (result == NULL) err("unable to allocate MPI result buffer");
    next_task = 0;
    active_workers = 0;

    /* Initial dispatch: leave at least one frequency for rank 0 when work
     * exists, so the master does not sit idle while workers solve.
     */
    initial_worker_limit = (emf->nfreq > 0) ? emf->nfreq - 1 : 0;
    for (worker = 1; worker < size; ++worker) {
      if (next_task < initial_worker_limit) {
        MPI_Send(&next_task, 1, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
        ++next_task;
        ++active_workers;
      } else {
        MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
      }
    }

    while (active_workers > 0 || next_task < emf->nfreq) {
      flag = 0;
      if (active_workers > 0) {
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT_INDEX, MPI_COMM_WORLD, &flag, &status);
      }

      if (flag) {
        MPI_Recv(&result_ifreq, 1, MPI_INT, status.MPI_SOURCE, TAG_RESULT_INDEX, MPI_COMM_WORLD, &status);
        worker = status.MPI_SOURCE;
        MPI_Recv(result, 4 * acq->nrec * (int)sizeof(float _Complex), MPI_BYTE, worker, TAG_RESULT_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        store_frequency_result(acq, emf, result_ifreq, result);

        if (emf->verb) printf("rank 0 collected freq=%g from worker %d\n", emf->freqs[result_ifreq], worker);

        if (next_task < emf->nfreq) {
          MPI_Send(&next_task, 1, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
          ++next_task;
        } else {
          MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
          --active_workers;
        }
        continue;
      }

      if (next_task < emf->nfreq) {
        int local_ifreq = next_task;
        ++next_task;
        if (emf->verb) printf("rank 0 processing modelling freq=%g locally\n", emf->freqs[local_ifreq]);
        solve_frequency(acq, emf, local_ifreq, local_ifreq, NULL);
        continue;
      }

      if (active_workers > 0) {
        MPI_Recv(&result_ifreq, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT_INDEX, MPI_COMM_WORLD, &status);
        worker = status.MPI_SOURCE;
        MPI_Recv(result, 4 * acq->nrec * (int)sizeof(float _Complex), MPI_BYTE, worker, TAG_RESULT_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        store_frequency_result(acq, emf, result_ifreq, result);

        if (emf->verb) printf("rank 0 collected freq=%g from worker %d\n", emf->freqs[result_ifreq], worker);

        if (next_task < emf->nfreq) {
          MPI_Send(&next_task, 1, MPI_INT, worker, TAG_WORK, MPI_COMM_WORLD);
          ++next_task;
        } else {
          MPI_Send(NULL, 0, MPI_INT, worker, TAG_STOP, MPI_COMM_WORLD);
          --active_workers;
        }
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
    allocate_modelling_buffers(acq, emf, 1, 1);
    result = alloc1complexf(4 * acq->nrec);
    if (result == NULL) err("unable to allocate MPI worker result buffer");
    while (1) {
      /* Wait for either:
       * TAG_WORK: solve the assigned frequency and send back 4 impedance arrays
       * TAG_STOP: no more work remains, so exit cleanly
       */
      MPI_Recv(&assigned_ifreq, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      /* TAG_STOP exits the worker loop immediately, so the solve/send path below only runs for real work messages. */
      if (status.MPI_TAG == TAG_STOP) break;
      if (status.MPI_TAG != TAG_WORK)
        err("worker received unexpected modelling MPI tag %d", status.MPI_TAG);

      /* solve_frequency() fills the worker-local field buffers, forms the
       * impedance tensor for one frequency, and packs Zxx/Zxy/Zyx/Zyy into
       * result[] for transmission back to rank 0.
       */
      solve_frequency(acq, emf, assigned_ifreq, 0, result);
      MPI_Send(&assigned_ifreq, 1, MPI_INT, 0, TAG_RESULT_INDEX, MPI_COMM_WORLD);
      MPI_Send(result, 4 * acq->nrec * (int)sizeof(float _Complex), MPI_BYTE, 0, TAG_RESULT_DATA, MPI_COMM_WORLD);
    }
    free1complexf(result);
    free_modelling_buffers(emf);
  }
}
