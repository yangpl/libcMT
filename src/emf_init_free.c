/* Electromagnetic-model initialization and teardown.
 * Loads frequencies and anisotropic resistivity models from HDF5, derives
 * mesh statistics and resistivity bounds, and releases the model storage.
 *----------------------------------------------------------------------
 *   Copyright (c) Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *----------------------------------------------------------------------
 */
#include "cstd.h"
#include "emf.h"
#include <hdf5.h>

static void check_h5(herr_t status, const char *message)
{
  if(status < 0) err("%s", message);
}

static void read_hdf5_vector(hid_t file_id, const char *name, int *n, float **data)
{
  hid_t dataset_id, space_id;
  hsize_t dims[1];

  dataset_id = H5Dopen2(file_id, name, H5P_DEFAULT);
  if(dataset_id < 0) err("cannot open HDF5 dataset %s", name);
  space_id = H5Dget_space(dataset_id);
  if(space_id < 0) err("cannot get HDF5 dataspace for %s", name);
  if(H5Sget_simple_extent_ndims(space_id) != 1) err("dataset %s must be 1D", name);
  check_h5(H5Sget_simple_extent_dims(space_id, dims, NULL), "cannot read HDF5 vector dimensions");

  *n = (int)dims[0];
  *data = alloc1float(*n);
  if(*data == NULL) err("cannot allocate vector for %s", name);
  check_h5(H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *data),
           "cannot read HDF5 vector");

  check_h5(H5Sclose(space_id), "cannot close HDF5 dataspace");
  check_h5(H5Dclose(dataset_id), "cannot close HDF5 dataset");
}

static void read_hdf5_model(hid_t file_id, const char *name, int nx, int ny, int nz, float ****data)
{
  hid_t dataset_id, space_id;
  hsize_t dims[3];

  dataset_id = H5Dopen2(file_id, name, H5P_DEFAULT);
  if(dataset_id < 0) err("cannot open HDF5 dataset %s", name);
  space_id = H5Dget_space(dataset_id);
  if(space_id < 0) err("cannot get HDF5 dataspace for %s", name);
  if(H5Sget_simple_extent_ndims(space_id) != 3) err("dataset %s must be 3D", name);
  check_h5(H5Sget_simple_extent_dims(space_id, dims, NULL), "cannot read HDF5 model dimensions");

  if((int)dims[0] != nz || (int)dims[1] != ny || (int)dims[2] != nx)
    err("dataset %s has shape [%d,%d,%d], expected [%d,%d,%d]",
        name, (int)dims[2], (int)dims[1], (int)dims[0], nx, ny, nz);

  *data = alloc3float(nx, ny, nz);
  if(*data == NULL) err("cannot allocate model for %s", name);
  check_h5(H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(*data)[0][0][0]),
           "cannot read HDF5 model");

  check_h5(H5Sclose(space_id), "cannot close HDF5 dataspace");
  check_h5(H5Dclose(dataset_id), "cannot close HDF5 dataset");
}

static void read_frequencies(emf_t *emf)
{
  char *ffreqs;
  hid_t file_id;

  if((emf->nfreq = countparval("freqs")) > 0) {
    emf->freqs = alloc1float(emf->nfreq);
    getparfloat("freqs", emf->freqs);
    return;
  }

  if(!getparstring("ffreqs", &ffreqs)) err("Need freqs= vector or ffreqs= HDF5 file");

  file_id = H5Fopen(ffreqs, H5F_ACC_RDONLY, H5P_DEFAULT);
  if(file_id < 0) err("cannot open HDF5 frequency file ffreqs=%s", ffreqs);
  read_hdf5_vector(file_id, "freqs", &emf->nfreq, &emf->freqs);
  check_h5(H5Fclose(file_id), "cannot close HDF5 frequency file");
}

void emf_init(emf_t *emf)
{
  int i1, i2, i3, ifreq;
  int found_noair;
  int n1node, n2node, n3node;
  char *fmodel;
  hid_t file_id;

  if(emf->verb) printf("----------- emf_init -------\n");
  if(!getparfloat("tol", &emf->tol)) emf->tol = 1e-7;/* stopping criteria */

  /* Frequencies can come either from a command-line freqs= list or from an HDF5
   * file/dataset pair using ffreq= and optional dfreq=. */
  read_frequencies(emf);
  emf->omegas = alloc1float(emf->nfreq);
  if(emf->verb) {
    printf("freqs=");
    for(ifreq = 0; ifreq < emf->nfreq; ++ifreq) {
      emf->omegas[ifreq] = 2.0f * PI * emf->freqs[ifreq];
      printf("%g,", emf->freqs[ifreq]);
    }
    printf("\n");
  } else {
    for(ifreq = 0; ifreq < emf->nfreq; ++ifreq) emf->omegas[ifreq] = 2.0f * PI * emf->freqs[ifreq];
  }

  if(!(getparstring("fmodel", &fmodel))) err("Need fmodel= ");
  file_id = H5Fopen(fmodel, H5F_ACC_RDONLY, H5P_DEFAULT);
  if(file_id < 0) err("cannot open HDF5 file fmodel=%s", fmodel);

  read_hdf5_vector(file_id, "fx1", &n1node, &emf->x1node);
  read_hdf5_vector(file_id, "fx2", &n2node, &emf->x2node);
  read_hdf5_vector(file_id, "fx3", &n3node, &emf->x3node);
  emf->nx = n1node - 1;
  emf->ny = n2node - 1;
  emf->nz = n3node - 1;

  read_hdf5_model(file_id, "frho11", emf->nx, emf->ny, emf->nz, &emf->rho11);
  read_hdf5_model(file_id, "frho22", emf->nx, emf->ny, emf->nz, &emf->rho22);
  read_hdf5_model(file_id, "frho33", emf->nx, emf->ny, emf->nz, &emf->rho33);
  check_h5(H5Fclose(file_id), "cannot close HDF5 model file");

  emf->x1min = emf->x1node[0];
  emf->x1max = emf->x1node[emf->nx];
  emf->x2min = emf->x2node[0];
  emf->x2max = emf->x2node[emf->ny];
  emf->x3min = emf->x3node[0];
  emf->x3max = emf->x3node[emf->nz];
  emf->d1min = emf->x1node[1] - emf->x1node[0];
  for(i1 = 0; i1 < emf->nx; i1++) emf->d1min = MIN(emf->d1min, emf->x1node[i1 + 1] - emf->x1node[i1]);
  emf->d2min = emf->x2node[1] - emf->x2node[0];
  for(i2 = 0; i2 < emf->ny; i2++) emf->d2min = MIN(emf->d2min, emf->x2node[i2 + 1] - emf->x2node[i2]);
  emf->d3min = emf->x3node[1] - emf->x3node[0];
  for(i3 = 0; i3 < emf->nz; i3++) emf->d3min = MIN(emf->d3min, emf->x3node[i3 + 1] - emf->x3node[i3]);
  if(emf->verb) {
    printf("[nx, ny, nz]=[%d, %d, %d]\n", emf->nx, emf->ny, emf->nz);
    printf("model domain [x1min, x1max]=[%g, %g]\n", emf->x1node[0], emf->x1node[emf->nx]);
    printf("model domain [x2min, x2max]=[%g, %g]\n", emf->x2node[0], emf->x2node[emf->ny]);
    printf("model domain [x3min, x3max]=[%g, %g]\n", emf->x3node[0], emf->x3node[emf->nz]);
    printf("minimum gridsize [d1min, d2min, d3min]=[%g, %g, %g]\n", emf->d1min, emf->d2min, emf->d3min);
  }

  emf->rhomax = emf->rho11[0][0][0];
  emf->rhomin = emf->rho11[0][0][0];
  for(i3 = 0; i3 < emf->nz; i3++) {
    for(i2 = 0; i2 < emf->ny; i2++) {
      for(i1 = 0; i1 < emf->nx; i1++) {
        emf->rhomax = MAX(emf->rhomax, emf->rho11[i3][i2][i1]);
        emf->rhomin = MIN(emf->rhomin, emf->rho11[i3][i2][i1]);
        emf->rhomax = MAX(emf->rhomax, emf->rho22[i3][i2][i1]);
        emf->rhomin = MIN(emf->rhomin, emf->rho22[i3][i2][i1]);
        emf->rhomax = MAX(emf->rhomax, emf->rho33[i3][i2][i1]);
        emf->rhomin = MIN(emf->rhomin, emf->rho33[i3][i2][i1]);
      }
    }
  }
  if(!getparfloat("rho_air", &emf->rho_air)) emf->rho_air = 1e8;/* resistivity threshold to be considered as air */
  emf->rhomax_noair = 0.0f;
  found_noair = 0;
  for(i3 = 0; i3 < emf->nz; i3++) {
    for(i2 = 0; i2 < emf->ny; i2++) {
      for(i1 = 0; i1 < emf->nx; i1++) {
        if(emf->rho11[i3][i2][i1] < emf->rho_air) {
          emf->rhomax_noair = found_noair ? MAX(emf->rhomax_noair, emf->rho11[i3][i2][i1]) : emf->rho11[i3][i2][i1];
          found_noair = 1;
        }
        if(emf->rho22[i3][i2][i1] < emf->rho_air) {
          emf->rhomax_noair = found_noair ? MAX(emf->rhomax_noair, emf->rho22[i3][i2][i1]) : emf->rho22[i3][i2][i1];
          found_noair = 1;
        }
        if(emf->rho33[i3][i2][i1] < emf->rho_air) {
          emf->rhomax_noair = found_noair ? MAX(emf->rhomax_noair, emf->rho33[i3][i2][i1]) : emf->rho33[i3][i2][i1];
          found_noair = 1;
        }
      }
    }
  }
  if(!found_noair) emf->rhomax_noair = emf->rhomax;
  if(emf->verb) {
    printf("[rhomin,rhomax]=[%g,%g]\n", emf->rhomin, emf->rhomax);
    if(!found_noair) printf("rhomax_noair=%g\n", emf->rhomax_noair);
  }
}

void emf_free(emf_t *emf)
{
  free1float(emf->freqs);
  free1float(emf->omegas);
  free1float(emf->x1node);
  free1float(emf->x2node);
  free1float(emf->x3node);
  free3float(emf->rho11);
  free3float(emf->rho22);
  free3float(emf->rho33);
}
