/* HDF5 I/O helpers for MT data files.
 * Reads observed impedance tensors for inversion and serializes
 * forward-model outputs into an output HDF5 file.
 *----------------------------------------------------------------------
 *   Copyright (c) Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *----------------------------------------------------------------------
 */
#include "cstd.h"
#include "acq.h"
#include "emf.h"
#include <hdf5.h>

static void check_hdf5_status(herr_t status, const char *message)
{
  if(status < 0) err("%s", message);
}

static void write_float_dataset_1d(hid_t file_id, const char *name, hsize_t n1, const float *data)
{
  hid_t space_id, dataset_id;
  hsize_t dims[1];

  dims[0] = n1;
  space_id = H5Screate_simple(1, dims, NULL);
  if(space_id < 0) err("error creating HDF5 dataspace");
  dataset_id = H5Dcreate2(file_id, name, H5T_IEEE_F32LE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if(dataset_id < 0) err("error creating HDF5 dataset");
  check_hdf5_status(H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
		    "error writing HDF5 float dataset");
  check_hdf5_status(H5Dclose(dataset_id), "error closing HDF5 dataset");
  check_hdf5_status(H5Sclose(space_id), "error closing HDF5 dataspace");
}

static void write_int_dataset_1d(hid_t file_id, const char *name, hsize_t n1, const int *data)
{
  hid_t space_id, dataset_id;
  hsize_t dims[1];

  dims[0] = n1;
  space_id = H5Screate_simple(1, dims, NULL);
  if(space_id < 0) err("error creating HDF5 dataspace");
  dataset_id = H5Dcreate2(file_id, name, H5T_STD_I32LE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if(dataset_id < 0) err("error creating HDF5 dataset");
  check_hdf5_status(H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
		    "error writing HDF5 int dataset");
  check_hdf5_status(H5Dclose(dataset_id), "error closing HDF5 dataset");
  check_hdf5_status(H5Sclose(space_id), "error closing HDF5 dataspace");
}

static void write_float_dataset_2d(hid_t file_id, const char *name, hsize_t n1, hsize_t n2, const float *data)
{
  hid_t space_id, dataset_id;
  hsize_t dims[2];

  dims[0] = n1;
  dims[1] = n2;
  space_id = H5Screate_simple(2, dims, NULL);
  if(space_id < 0) err("error creating HDF5 dataspace");
  dataset_id = H5Dcreate2(file_id, name, H5T_IEEE_F32LE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if(dataset_id < 0) err("error creating HDF5 dataset");
  check_hdf5_status(H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
		    "error writing HDF5 float dataset");
  check_hdf5_status(H5Dclose(dataset_id), "error closing HDF5 dataset");
  check_hdf5_status(H5Sclose(space_id), "error closing HDF5 dataspace");
}

/* Store complex MT tensors as [nfreq, nrec, 2] with real/imag parts in the last axis. */
static void write_complex_dataset_3d(hid_t file_id, const char *name, int nfreq, int nrec, float _Complex **values)
{
  hid_t space_id, dataset_id;
  hsize_t dims[3];
  float *data;
  int ifreq, irec;
  size_t idx;

  dims[0] = nfreq;
  dims[1] = nrec;
  dims[2] = 2;
  data = alloc1float((size_t)nfreq * nrec * 2);
  if(data == NULL) err("error allocating HDF5 output buffer");

  /* Flatten the complex array into an HDF5-friendly real buffer. */
  idx = 0;
  for(ifreq=0; ifreq<nfreq; ifreq++){
    for(irec=0; irec<nrec; irec++){
      data[idx++] = crealf(values[ifreq][irec]);
      data[idx++] = cimagf(values[ifreq][irec]);
    }
  }

  space_id = H5Screate_simple(3, dims, NULL);
  if(space_id < 0) err("error creating HDF5 dataspace");
  dataset_id = H5Dcreate2(file_id, name, H5T_IEEE_F32LE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if(dataset_id < 0) err("error creating HDF5 dataset");
  check_hdf5_status(H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
		    "error writing HDF5 complex dataset");
  check_hdf5_status(H5Dclose(dataset_id), "error closing HDF5 dataset");
  check_hdf5_status(H5Sclose(space_id), "error closing HDF5 dataspace");
  free1float(data);
}

/* Read a [nfreq, nrec, 2] tensor back into the solver's complex storage. */
static void read_complex_dataset_3d(hid_t file_id, const char *name,
                                    int nfreq, int nrec, float _Complex **values)
{
  hid_t dataset_id, space_id;
  hsize_t dims[3];
  float *data;
  int ifreq, irec;
  size_t idx;

  dataset_id = H5Dopen2(file_id, name, H5P_DEFAULT);
  if(dataset_id < 0) err("cannot open HDF5 dataset %s", name);
  space_id = H5Dget_space(dataset_id);
  if(space_id < 0) err("cannot get HDF5 dataspace for %s", name);
  if(H5Sget_simple_extent_ndims(space_id) != 3) err("dataset %s must be 3D", name);
  check_hdf5_status(H5Sget_simple_extent_dims(space_id, dims, NULL), "cannot read HDF5 dataset dimensions");
  if((int)dims[0] != nfreq || (int)dims[1] != nrec || (int)dims[2] != 2) {
    err("dataset %s has shape [%d,%d,%d], expected [%d,%d,2]",
        name, (int)dims[0], (int)dims[1], (int)dims[2], nfreq, nrec);
  }

  data = alloc1float((size_t)nfreq * nrec * 2);
  if(data == NULL) err("cannot allocate buffer for %s", name);
  check_hdf5_status(H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
                    "cannot read HDF5 complex dataset");

  /* Reconstruct complex values from the trailing real/imag dimension. */
  idx = 0;
  for(ifreq = 0; ifreq < nfreq; ++ifreq) {
    for(irec = 0; irec < nrec; ++irec) {
      values[ifreq][irec] = data[idx] + I * data[idx + 1];
      idx += 2;
    }
  }

  free1float(data);
  check_hdf5_status(H5Sclose(space_id), "cannot close HDF5 dataspace");
  check_hdf5_status(H5Dclose(dataset_id), "cannot close HDF5 dataset");
}

/* Load observed impedance tensors for all frequencies and receivers. */
void read_mt_data(acq_t *acq, emf_t *emf, char *fname)
{
  hid_t file_id;

  file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  if(file_id < 0) err("cannot open observed MT data file %s", fname);

  read_complex_dataset_3d(file_id, "Zxx", emf->nfreq, acq->nrec, emf->obs_Zxx);
  read_complex_dataset_3d(file_id, "Zxy", emf->nfreq, acq->nrec, emf->obs_Zxy);
  read_complex_dataset_3d(file_id, "Zyx", emf->nfreq, acq->nrec, emf->obs_Zyx);
  read_complex_dataset_3d(file_id, "Zyy", emf->nfreq, acq->nrec, emf->obs_Zyy);

  check_hdf5_status(H5Fclose(file_id), "cannot close observed MT data file");
}

/* Write modelled MT responses together with the receiver metadata used to sample them. */
void write_mt_data(acq_t *acq, emf_t *emf, char *fname)
{
  hid_t file_id;
  float *receiver_position;
  int *receiver_index;
  int irec, ifreq;
  size_t idx;

  receiver_index = alloc1int(acq->nrec);
  receiver_position = alloc1float((size_t)acq->nrec * 3);
  if(receiver_index == NULL || receiver_position == NULL) err("error allocating HDF5 receiver buffers");

  /* Flatten receiver coordinates so they can be written as an [nrec, 3] dataset. */
  idx = 0;
  for(irec=0; irec<acq->nrec; irec++){
    receiver_index[irec] = irec + 1;
    receiver_position[idx++] = acq->rec_x1[irec];
    receiver_position[idx++] = acq->rec_x2[irec];
    receiver_position[idx++] = acq->rec_x3[irec];
  }
  for(ifreq=0; ifreq<emf->nfreq; ifreq++){
    if(!isfinite(emf->freqs[ifreq])) err("invalid MT frequency encountered before HDF5 write");
  }

  file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(file_id < 0) err("error opening MT output HDF5 file for writing");

  write_float_dataset_1d(file_id, "frequencies", emf->nfreq, emf->freqs);
  write_int_dataset_1d(file_id, "receiver_index", acq->nrec, receiver_index);
  write_float_dataset_2d(file_id, "receiver_position", acq->nrec, 3, receiver_position);
  write_complex_dataset_3d(file_id, "Zxx", emf->nfreq, acq->nrec, emf->cal_Zxx);
  write_complex_dataset_3d(file_id, "Zxy", emf->nfreq, acq->nrec, emf->cal_Zxy);
  write_complex_dataset_3d(file_id, "Zyx", emf->nfreq, acq->nrec, emf->cal_Zyx);
  write_complex_dataset_3d(file_id, "Zyy", emf->nfreq, acq->nrec, emf->cal_Zyy);

  check_hdf5_status(H5Fclose(file_id), "error closing MT output HDF5 file");
  free1int(receiver_index);
  free1float(receiver_position);
}

static void write_float_dataset_3d(hid_t file_id, const char *name,
                                   hsize_t n1, hsize_t n2, hsize_t n3, const float *data)
{
  hid_t space_id, dataset_id;
  hsize_t dims[3];

  dims[0] = n1;
  dims[1] = n2;
  dims[2] = n3;
  space_id = H5Screate_simple(3, dims, NULL);
  if(space_id < 0) err("error creating HDF5 dataspace");
  dataset_id = H5Dcreate2(file_id, name, H5T_IEEE_F32LE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if(dataset_id < 0) err("error creating HDF5 dataset");
  check_hdf5_status(H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
                    "error writing HDF5 float dataset");
  check_hdf5_status(H5Dclose(dataset_id), "error closing HDF5 dataset");
  check_hdf5_status(H5Sclose(space_id), "error closing HDF5 dataspace");
}

void write_computational_model_hdf5(emf_t *emf, int ifreq, const char *fname)
{
  hid_t file_id;
  float *x1, *x2, *x3;
  float *rho11, *rho22, *rho33;
  float frequency;
  int i1, i2, i3;
  size_t id;
  size_t ncell;

  if(emf == NULL || fname == NULL) err("invalid computational mesh output request");
  if(emf->x1 == NULL || emf->x2 == NULL || emf->x3 == NULL ||
     emf->sigma11 == NULL || emf->sigma22 == NULL || emf->sigma33 == NULL)
    err("computational mesh output requested before extended model is initialized");

  ncell = (size_t)emf->n1 * emf->n2 * emf->n3;
  x1 = alloc1float(emf->n1 + 1);
  x2 = alloc1float(emf->n2 + 1);
  x3 = alloc1float(emf->n3 + 1);
  rho11 = alloc1float(ncell);
  rho22 = alloc1float(ncell);
  rho33 = alloc1float(ncell);
  if(x1 == NULL || x2 == NULL || x3 == NULL ||
     rho11 == NULL || rho22 == NULL || rho33 == NULL)
    err("error allocating computational mesh output buffers");

  for(i1 = 0; i1 <= emf->n1; ++i1) x1[i1] = (float)emf->x1[i1];
  for(i2 = 0; i2 <= emf->n2; ++i2) x2[i2] = (float)emf->x2[i2];
  for(i3 = 0; i3 <= emf->n3; ++i3) x3[i3] = (float)emf->x3[i3];

  id = 0;
  for(i3 = 0; i3 < emf->n3; ++i3) {
    for(i2 = 0; i2 < emf->n2; ++i2) {
      for(i1 = 0; i1 < emf->n1; ++i1) {
        if(emf->sigma11[i3][i2][i1] <= 0.0 ||
           emf->sigma22[i3][i2][i1] <= 0.0 ||
           emf->sigma33[i3][i2][i1] <= 0.0)
          err("nonpositive conductivity in computational mesh output");
        rho11[id] = (float)(1.0 / emf->sigma11[i3][i2][i1]);
        rho22[id] = (float)(1.0 / emf->sigma22[i3][i2][i1]);
        rho33[id] = (float)(1.0 / emf->sigma33[i3][i2][i1]);
        ++id;
      }
    }
  }

  file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(file_id < 0) err("error opening computational mesh HDF5 file for writing");

  write_float_dataset_1d(file_id, "fx1", emf->n1 + 1, x1);
  write_float_dataset_1d(file_id, "fx2", emf->n2 + 1, x2);
  write_float_dataset_1d(file_id, "fx3", emf->n3 + 1, x3);
  write_float_dataset_3d(file_id, "frho11", emf->n3, emf->n2, emf->n1, rho11);
  write_float_dataset_3d(file_id, "frho22", emf->n3, emf->n2, emf->n1, rho22);
  write_float_dataset_3d(file_id, "frho33", emf->n3, emf->n2, emf->n1, rho33);
  if(ifreq >= 0 && ifreq < emf->nfreq) {
    frequency = emf->freqs[ifreq];
    write_float_dataset_1d(file_id, "frequency", 1, &frequency);
  }

  check_hdf5_status(H5Fclose(file_id), "error closing computational mesh HDF5 file");
  free1float(x1);
  free1float(x2);
  free1float(x3);
  free1float(rho11);
  free1float(rho22);
  free1float(rho33);
}

void write_inversion_model_hdf5(emf_t *emf, const float *x)
{
  hid_t file_id;
  float *rho11, *rho22, *rho33;
  int i, j, k;
  int id;
  int ncell;

  if(emf == NULL || x == NULL) err("invalid inversion snapshot request");

  ncell = emf->nx * emf->ny * emf->nz;
  rho11 = alloc1float(ncell);
  rho22 = alloc1float(ncell);
  rho33 = alloc1float(ncell);
  if(rho11 == NULL || rho22 == NULL || rho33 == NULL)
    err("error allocating inversion snapshot buffers");

  for(k = 0; k < emf->nz; ++k) {
    for(j = 0; j < emf->ny; ++j) {
      for(i = 0; i < emf->nx; ++i) {
        id = i + emf->nx * (j + emf->ny * k);
        rho11[id] = expf(-x[id]);
        rho22[id] = expf(-x[id]);
        rho33[id] = expf(-x[id + ncell]);
      }
    }
  }

  file_id = H5Fcreate("model_final.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(file_id < 0) err("error opening inversion snapshot HDF5 file for writing");

  /* Keep the full solver input contract so model_final.h5 can be used
   * directly as fmodel= when restarting inversion. */
  write_float_dataset_1d(file_id, "fx1", emf->nx + 1, emf->x1node);
  write_float_dataset_1d(file_id, "fx2", emf->ny + 1, emf->x2node);
  write_float_dataset_1d(file_id, "fx3", emf->nz + 1, emf->x3node);
  write_float_dataset_3d(file_id, "frho11", emf->nz, emf->ny, emf->nx, rho11);
  write_float_dataset_3d(file_id, "frho22", emf->nz, emf->ny, emf->nx, rho22);
  write_float_dataset_3d(file_id, "frho33", emf->nz, emf->ny, emf->nx, rho33);
  /* Keep the reduced VTI view for post-processing scripts that already use it. */
  write_float_dataset_3d(file_id, "rho_h", emf->nz, emf->ny, emf->nx, rho11);
  write_float_dataset_3d(file_id, "rho_v", emf->nz, emf->ny, emf->nx, rho33);

  check_hdf5_status(H5Fclose(file_id), "error closing inversion snapshot HDF5 file");
  free1float(rho11);
  free1float(rho22);
  free1float(rho33);
}

void write_inversion_gradient_hdf5(emf_t *emf, const float *g)
{
  hid_t file_id;
  int ncell;

  if(emf == NULL || g == NULL) err("invalid inversion gradient snapshot request");

  ncell = emf->nx * emf->ny * emf->nz;
  file_id = H5Fcreate("gradient.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(file_id < 0) err("error opening inversion gradient HDF5 file for writing");

  write_float_dataset_1d(file_id, "fx1", emf->nx + 1, emf->x1node);
  write_float_dataset_1d(file_id, "fx2", emf->ny + 1, emf->x2node);
  write_float_dataset_1d(file_id, "fx3", emf->nz + 1, emf->x3node);
  write_float_dataset_3d(file_id, "grad_mh", emf->nz, emf->ny, emf->nx, g);
  write_float_dataset_3d(file_id, "grad_mv", emf->nz, emf->ny, emf->nx, g + ncell);

  check_hdf5_status(H5Fclose(file_id), "error closing inversion gradient HDF5 file");
}
