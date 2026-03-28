/* Electromagnetic model and data-state definitions.
 * Declares the main container holding frequencies, resistivity models,
 * computational-grid properties, and forward/adjoint field arrays.
 */
#ifndef emf_h
#define emf_h

#pragma once

typedef struct {
  int mode; //mode=0, modeling; mode=1, FWI; mode=2, FWI gradient only
  int verb;/* verbose display */
  
  int nfreq;//number of frequencies
  float *freqs, *omegas;//a list of frequencies
  float rho_air;//resistivity in air
  float lextend;//domain extension
  float rhomin, rhomax, rhomax_noair;
  
  int nx, ny, nz;//size of the model defined on regular FD grid
  float *x1node, *x2node, *x3node;
  float ***rho11, ***rho22, ***rho33;
  double d1min, d2min, d3min;
  double x1min, x2min, x3min;
  double x1max, x2max, x3max;

  int n1, n2, n3;
  double *x1, *x2, *x3;
  double ***sigma11, ***sigma22, ***sigma33, ***invmur;

  float _Complex ***Efwd, ***Eadj;//forward and adjoint fields
  float _Complex ***d_Ex, ***d_Ey, ***d_Hx, ***d_Hy;//calculated data
  float _Complex **obs_Zxx, **obs_Zyx, **obs_Zxy, **obs_Zyy;
  float _Complex **cal_Zxx, **cal_Zyx, **cal_Zxy, **cal_Zyy;
  float _Complex **res_Zxx, **res_Zyx, **res_Zxy, **res_Zyy;
  float _Complex ***s_Ex, ***s_Ey, ***s_Hx, ***s_Hy;//adjoint sources

  float tol;
} emf_t; /* type of electromagnetic field (emf)  */

#endif
