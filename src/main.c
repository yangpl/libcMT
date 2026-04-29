/* Main entry point for the MT application.
 * Parses runtime arguments, initializes the model and acquisition state,
 * and dispatches either forward modelling or inversion mode.
 *----------------------------------------------------------------------
 *   Copyright (c) Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *----------------------------------------------------------------------
 */
#include "cstd.h"
#include "emf.h"
#include "acq.h"
#include <mpi.h>

void emf_init(emf_t *emf);
void emf_free(emf_t *emf);

void acq_init(acq_t *acq, emf_t *emf);
void acq_free(acq_t *acq);

void do_modelling(acq_t *acq, emf_t *emf);
int do_inversion(acq_t *acq, emf_t *emf);

int main(int argc, char **argv)
{
  emf_t *emf;
  acq_t *acq;
  int status = EXIT_SUCCESS;
  char current_time[128];
  time_t t;
  struct tm *ptm;
  int mpi_rank = 0;
  int mpi_size = 1;

  /* When stdout/stderr are redirected to log files, force line-buffered output so
   * long forward/adjoint solves still emit progress messages immediately. */
  setvbuf(stdout, NULL, _IOLBF, 0);
  setvbuf(stderr, NULL, _IOLBF, 0);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  initargs(argc, argv);

  acq = calloc(1, sizeof(acq_t));
  emf = calloc(1, sizeof(emf_t));
  if(acq == NULL || emf == NULL) err("failed to allocate application state");
  if(!getparint("verb", &emf->verb)) emf->verb = 1;/* 1 prints progress messages, 0 runs quietly. */
  if(!getparint("mode", &emf->mode)) emf->mode = 0;/* 0 = modelling, 1 = inversion, 2 = gradient only. */
  if(mpi_rank != 0) emf->verb = 0;

  if(emf->verb){
    t = time(NULL);
    ptm = localtime(&t);
    strftime(current_time, sizeof(current_time), "%d-%b-%Y %H:%M:%S", ptm);
    printf("  Current date and time: %s\n", current_time);
    printf("=====================================================\n");
    printf(" Welcome to libcMT for 3D MT modelling and inversion \n");
    printf("       ------------------------------------          \n");
    printf("            Copyright (c) Pengliang Yang             \n");
    printf("             Email: ypl.2100@gmail.com               \n");
    printf("=====================================================\n");
    if(emf->mode==0) printf("Task: MT modelling\n");
    if(emf->mode==1) printf("Task: MT inversion\n");
    if(emf->mode==2) printf("Task: Output inversion gradient\n");
  }

  emf_init(emf);
  acq_init(acq, emf);
  if(emf->mode==0) {
    do_modelling(acq, emf);
  } else if(emf->mode==1 || emf->mode==2) {
    status = do_inversion(acq, emf);
  } else {
    err("unknown mode=%d; expected 0 modelling, 1 inversion, or 2 gradient only", emf->mode);
  }
  acq_free(acq);
  emf_free(emf);

  free(emf);
  free(acq);
  MPI_Finalize();

  return status;
}
