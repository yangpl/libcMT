/* Public interface for the geometric multigrid solver.
 * Exposes global solver state, grid-level storage, and the routines used
 * to build, apply, and destroy the multigrid hierarchy.
 */
#ifndef GMG_H
#define GMG_H

#pragma once

#include "emf.h"

extern int verb;
extern int cycleopt;
extern int icycle, ncycle;
extern int isemicoarsen;
extern int v1;
extern int v2;
extern int lmax;
extern double tol;
extern double rnorm0, rnorm;
extern complex I_omega_mu0;
extern emf_t *emf_;

typedef struct {
  int n1, n2, n3;
  double *x1, *x2, *x3;
  double *x1s, *x2s, *x3s;
  double *d1, *d2, *d3;
  double *d1s, *d2s, *d3s;
  complex ****u, ****f, ****r;
  double ***sigma11, ***sigma22, ***sigma33;
  double ***invmur;
  int sc[3];
} gmg_t;

extern gmg_t *gmg;

void residual(gmg_t *gmg, int lev);
void compute_H_from_E(gmg_t *gmg);
void v_cycle(gmg_t *gmg, int lev);
void f_cycle(gmg_t *gmg, int lev);
void grid_init(gmg_t *gmg, int lev);
void grid_free(gmg_t *gmg, int lev);
void gmg_init(emf_t *emf, int ifreq);
void gmg_free(void);


#endif
