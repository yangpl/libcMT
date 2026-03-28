/* Acquisition-geometry initialization and teardown.
 * Reads MT receiver positions and orientations from HDF5, validates them
 * against the model bounds, and stores the survey definition in acq_t.
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

static void check_h5(herr_t status, const char *message)
{
  if(status < 0) err("%s", message);
}

static void read_hdf5_receiver_positions(hid_t file_id, int *nrec, float **data)
{
  hid_t dataset_id, space_id;
  hsize_t dims[2];

  dataset_id = H5Dopen2(file_id, "receiver_position", H5P_DEFAULT);
  if(dataset_id < 0) err("cannot open HDF5 dataset receiver_position");
  space_id = H5Dget_space(dataset_id);
  if(space_id < 0) err("cannot get HDF5 dataspace for receiver_position");
  if(H5Sget_simple_extent_ndims(space_id) != 2) err("receiver_position must be 2D");
  check_h5(H5Sget_simple_extent_dims(space_id, dims, NULL), "cannot read receiver_position dimensions");
  if((int)dims[1] != 3) err("receiver_position must have shape [nrec,3]");

  *nrec = (int)dims[0];
  *data = alloc1float((size_t)(*nrec) * 3);
  if(*data == NULL) err("cannot allocate receiver_position");
  check_h5(H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *data),
           "cannot read receiver_position");

  check_h5(H5Sclose(space_id), "cannot close HDF5 dataspace");
  check_h5(H5Dclose(dataset_id), "cannot close HDF5 dataset");
}

static void read_hdf5_float_vector(hid_t file_id, const char *name, int n, float *data)
{
  hid_t dataset_id, space_id;
  hsize_t dims[1];

  dataset_id = H5Dopen2(file_id, name, H5P_DEFAULT);
  if(dataset_id < 0) err("cannot open HDF5 dataset %s", name);
  space_id = H5Dget_space(dataset_id);
  if(space_id < 0) err("cannot get HDF5 dataspace for %s", name);
  if(H5Sget_simple_extent_ndims(space_id) != 1) err("dataset %s must be 1D", name);
  check_h5(H5Sget_simple_extent_dims(space_id, dims, NULL), "cannot read HDF5 vector dimensions");
  if((int)dims[0] != n) err("dataset %s has wrong length", name);
  check_h5(H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
           "cannot read HDF5 float vector");

  check_h5(H5Sclose(space_id), "cannot close HDF5 dataspace");
  check_h5(H5Dclose(dataset_id), "cannot close HDF5 dataset");
}

/*< read receiver file to initialize MT survey geometry >*/
void acq_init(acq_t *acq, emf_t *emf)
{
  static int nd = 5000;
  float rec_x1[nd], rec_x2[nd], rec_x3[nd], rec_hd[nd], rec_pit[nd];
  float *receiver_position = NULL;
  float x1min, x1max, x2min, x2max, x3min, x3max;
  float model_x1min, model_x1max, model_x2min, model_x2max, model_x3min, model_x3max;
  int irec;
  char *frec;
  hid_t file_id;

  if(emf->verb) printf("---------acq_init ---------\n");

  model_x1min = emf->x1node[0];
  model_x1max = emf->x1node[emf->nx];
  model_x2min = emf->x2node[0];
  model_x2max = emf->x2node[emf->ny];
  model_x3min = emf->x3node[0];
  model_x3max = emf->x3node[emf->nz];

  if(!getparfloat("x1min", &acq->x1min)) acq->x1min = model_x1min;
  if(!getparfloat("x1max", &acq->x1max)) acq->x1max = model_x1max;
  if(!getparfloat("x2min", &acq->x2min)) acq->x2min = model_x2min;
  if(!getparfloat("x2max", &acq->x2max)) acq->x2max = model_x2max;
  if(!getparfloat("x3min", &acq->x3min)) acq->x3min = model_x3min;
  if(!getparfloat("x3max", &acq->x3max)) acq->x3max = model_x3max;

  if(model_x1min>acq->x1min || model_x1max < acq->x1max)
    err("x - receivers from acquisition file are out of domain");
  if(model_x2min>acq->x2min || model_x2max < acq->x2max)
    err("y - receivers from acquisition file are out of domain");
  if(model_x3min>acq->x3min || model_x3max < acq->x3max)
    err("z - receivers from acquisition file are out of domain");

  x1min = acq->x1min;
  x1max = acq->x1max;
  x2min = acq->x2min;
  x2max = acq->x2max;
  x3min = acq->x3min;
  x3max = acq->x3max;

  if(!(getparstring("frec", &frec))) err("Need frec= ");
  file_id = H5Fopen(frec, H5F_ACC_RDONLY, H5P_DEFAULT);
  if(file_id < 0) err("cannot open HDF5 receiver file frec=%s", frec);
  read_hdf5_receiver_positions(file_id, &acq->nrec, &receiver_position);
  read_hdf5_float_vector(file_id, "receiver_azimuth", acq->nrec, rec_hd);
  read_hdf5_float_vector(file_id, "receiver_dip", acq->nrec, rec_pit);
  check_h5(H5Fclose(file_id), "cannot close HDF5 receiver file");
  for(irec=0; irec<acq->nrec; ++irec){
    rec_x1[irec] = receiver_position[3*irec];
    rec_x2[irec] = receiver_position[3*irec + 1];
    rec_x3[irec] = receiver_position[3*irec + 2];
  }
  free1float(receiver_position);

  for(irec=0; irec<acq->nrec; ++irec){
    x1min = MIN(x1min, rec_x1[irec]);
    x1max = MAX(x1max, rec_x1[irec]);
    x2min = MIN(x2min, rec_x2[irec]);
    x2max = MAX(x2max, rec_x2[irec]);
    x3min = MIN(x3min, rec_x3[irec]);
    x3max = MAX(x3max, rec_x3[irec]);
  }

  if(x1min<acq->x1min) err("receiver location: x<x1min");
  if(x2min<acq->x2min) err("receiver location: y<x2min");
  if(x3min<acq->x3min) err("receiver location: z<x3min");
  if(x1max>acq->x1max) err("receiver location: x>x1max");
  if(x2max>acq->x2max) err("receiver location: y>x2max");
  if(x3max>acq->x3max) err("receiver location: z>x3max");

  acq->rec_x1 = alloc1float(acq->nrec);
  acq->rec_x2 = alloc1float(acq->nrec);
  acq->rec_x3 = alloc1float(acq->nrec);
  acq->rec_azimuth = alloc1float(acq->nrec);
  acq->rec_dip = alloc1float(acq->nrec);
  for(irec=0; irec<acq->nrec; ++irec){
    acq->rec_x1[irec] = rec_x1[irec];
    acq->rec_x2[irec] = rec_x2[irec];
    acq->rec_x3[irec] = rec_x3[irec];
    acq->rec_azimuth[irec] = PI*rec_hd[irec]/180.;
    acq->rec_dip[irec] = PI*rec_pit[irec]/180.;
  }

  if(emf->verb){
    printf("number of receivers: nrec=%d\n", acq->nrec);
    printf("original domain [x1min, x1max]=[%g, %g]\n", acq->x1min, acq->x1max);
    printf("original domain [x2min, x2max]=[%g, %g]\n", acq->x2min, acq->x2max);
    printf("original domain [x3min, x3max]=[%g, %g]\n", acq->x3min, acq->x3max);
  }
}

/*< free the allocated variables for acquisition >*/
void acq_free(acq_t *acq)
{
  free1float(acq->rec_x1);
  free1float(acq->rec_x2);
  free1float(acq->rec_x3);
  free1float(acq->rec_azimuth);
  free1float(acq->rec_dip);
}
