/* Acquisition-geometry data definitions.
 * Declares the receiver layout container used to store MT station
 * coordinates, orientations, and survey bounds.
 */
#ifndef __acq_h__
#define __acq_h__

#pragma once

typedef struct {
  int nrec;/* number of receivers */
  float *rec_x1, *rec_x2, *rec_x3, *rec_azimuth, *rec_dip;
  float x1min, x1max, x2min, x2max, x3min, x3max;/* coordinate bounds */
} acq_t;/* type of acquisition geometry */

#endif
