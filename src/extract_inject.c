/* Receiver interpolation and adjoint-source injection utilities.
 * Samples E/H fields from the multigrid staggered grid at receiver
 * locations and distributes adjoint source terms back onto the grid.
 *----------------------------------------------------------------------
 *   Copyright (c) Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *----------------------------------------------------------------------
 */
#include "cstd.h"
#include "acq.h"
#include "gmg.h"

int find_index(int n, double *x, double val);

void extract_mt_data(acq_t *acq, int ifreq, int ipolar, float _Complex ***E)
{
  int i, j, k, irec;
  int ip1, jp1, kp1;
  double w1, w2, w3;
  complex s;
  complex ***Ex = gmg[0].u[0];
  complex ***Ey = gmg[0].u[1];
  complex ***Ez = gmg[0].u[2];
  complex ***Hx = gmg[0].f[0];
  complex ***Hy = gmg[0].f[1];

  for (irec = 0; irec < acq->nrec; irec++) {
    i = find_index(gmg[0].n1, gmg[0].x1s, acq->rec_x1[irec]);
    j = find_index(gmg[0].n2 + 1, gmg[0].x2, acq->rec_x2[irec]);
    k = find_index(gmg[0].n3 + 1, gmg[0].x3, acq->rec_x3[irec]);
    ip1 = MIN(i + 1, gmg[0].n1 - 1);
    jp1 = MIN(j + 1, gmg[0].n2);
    kp1 = MIN(k + 1, gmg[0].n3);
    w1 = (acq->rec_x1[irec] - gmg[0].x1s[i]) / gmg[0].d1[ip1];
    w2 = (acq->rec_x2[irec] - gmg[0].x2[j]) / gmg[0].d2s[j];
    w3 = (acq->rec_x3[irec] - gmg[0].x3[k]) / gmg[0].d3s[k];
    s = 0;
    s += Ex[k][j][i] * (1. - w1) * (1. - w2) * (1. - w3);
    s += Ex[k][j][ip1] * w1 * (1. - w2) * (1. - w3);
    s += Ex[k][jp1][i] * (1. - w1) * w2 * (1. - w3);
    s += Ex[k][jp1][ip1] * w1 * w2 * (1. - w3);
    s += Ex[kp1][j][i] * (1. - w1) * (1. - w2) * w3;
    s += Ex[kp1][j][ip1] * w1 * (1. - w2) * w3;
    s += Ex[kp1][jp1][i] * (1. - w1) * w2 * w3;
    s += Ex[kp1][jp1][ip1] * w1 * w2 * w3;
    emf_->d_Ex[ipolar][ifreq][irec] = s;

    i = find_index(gmg[0].n1 + 1, gmg[0].x1, acq->rec_x1[irec]);
    j = find_index(gmg[0].n2, gmg[0].x2s, acq->rec_x2[irec]);
    k = find_index(gmg[0].n3 + 1, gmg[0].x3, acq->rec_x3[irec]);
    ip1 = MIN(i + 1, gmg[0].n1);
    jp1 = MIN(j + 1, gmg[0].n2 - 1);
    kp1 = MIN(k + 1, gmg[0].n3);
    w1 = (acq->rec_x1[irec] - gmg[0].x1[i]) / gmg[0].d1s[i];
    w2 = (acq->rec_x2[irec] - gmg[0].x2s[j]) / gmg[0].d2[jp1];
    w3 = (acq->rec_x3[irec] - gmg[0].x3[k]) / gmg[0].d3s[k];
    s = 0;
    s += Ey[k][j][i] * (1. - w1) * (1. - w2) * (1. - w3);
    s += Ey[k][j][ip1] * w1 * (1. - w2) * (1. - w3);
    s += Ey[k][jp1][i] * (1. - w1) * w2 * (1. - w3);
    s += Ey[k][jp1][ip1] * w1 * w2 * (1. - w3);
    s += Ey[kp1][j][i] * (1. - w1) * (1. - w2) * w3;
    s += Ey[kp1][j][ip1] * w1 * (1. - w2) * w3;
    s += Ey[kp1][jp1][i] * (1. - w1) * w2 * w3;
    s += Ey[kp1][jp1][ip1] * w1 * w2 * w3;
    emf_->d_Ey[ipolar][ifreq][irec] = s;

    i = find_index(gmg[0].n1 + 1, gmg[0].x1, acq->rec_x1[irec]);
    j = find_index(gmg[0].n2, gmg[0].x2s, acq->rec_x2[irec]);
    k = find_index(gmg[0].n3, gmg[0].x3s, acq->rec_x3[irec]);
    ip1 = MIN(i + 1, gmg[0].n1);
    jp1 = MIN(j + 1, gmg[0].n2 - 1);
    kp1 = MIN(k + 1, gmg[0].n3);
    w1 = (acq->rec_x1[irec] - gmg[0].x1[i]) / gmg[0].d1s[i];
    w2 = (acq->rec_x2[irec] - gmg[0].x2s[j]) / gmg[0].d2[jp1];
    w3 = (acq->rec_x3[irec] - gmg[0].x3s[k]) / gmg[0].d3[kp1];
    s = 0;
    s += Hx[k][j][i] * (1. - w1) * (1. - w2) * (1. - w3);
    s += Hx[k][j][ip1] * w1 * (1. - w2) * (1. - w3);
    s += Hx[k][jp1][i] * (1. - w1) * w2 * (1. - w3);
    s += Hx[k][jp1][ip1] * w1 * w2 * (1. - w3);
    s += Hx[kp1][j][i] * (1. - w1) * (1. - w2) * w3;
    s += Hx[kp1][j][ip1] * w1 * (1. - w2) * w3;
    s += Hx[kp1][jp1][i] * (1. - w1) * w2 * w3;
    s += Hx[kp1][jp1][ip1] * w1 * w2 * w3;
    emf_->d_Hx[ipolar][ifreq][irec] = s;

    i = find_index(gmg[0].n1, gmg[0].x1s, acq->rec_x1[irec]);
    j = find_index(gmg[0].n2 + 1, gmg[0].x2, acq->rec_x2[irec]);
    k = find_index(gmg[0].n3, gmg[0].x3s, acq->rec_x3[irec]);
    ip1 = MIN(i + 1, gmg[0].n1 - 1);
    jp1 = MIN(j + 1, gmg[0].n2);
    kp1 = MIN(k + 1, gmg[0].n3 - 1);
    w1 = (acq->rec_x1[irec] - gmg[0].x1s[i]) / gmg[0].d1[ip1];
    w2 = (acq->rec_x2[irec] - gmg[0].x2[j]) / gmg[0].d2s[j];
    w3 = (acq->rec_x3[irec] - gmg[0].x3s[k]) / gmg[0].d3[kp1];
    s = 0;
    s += Hy[k][j][i] * (1. - w1) * (1. - w2) * (1. - w3);
    s += Hy[k][j][ip1] * w1 * (1. - w2) * (1. - w3);
    s += Hy[k][jp1][i] * (1. - w1) * w2 * (1. - w3);
    s += Hy[k][jp1][ip1] * w1 * w2 * (1. - w3);
    s += Hy[kp1][j][i] * (1. - w1) * (1. - w2) * w3;
    s += Hy[kp1][j][ip1] * w1 * (1. - w2) * w3;
    s += Hy[kp1][jp1][i] * (1. - w1) * w2 * w3;
    s += Hy[kp1][jp1][ip1] * w1 * w2 * w3;
    emf_->d_Hy[ipolar][ifreq][irec] = s;
  }

  if(E == NULL) return;

  int i1, i2, i3;
  int kk = 0;
  for(i3=0; i3<emf_->nz; i3++){
    k = find_index(gmg[0].n3+1, gmg[0].x3, emf_->x3[i3]);
    kp1 = MIN(k + 1, gmg[0].n3);
    w3 = (emf_->x3[i3] - gmg[0].x3[k]) / gmg[0].d3s[k];
    for(i2=0; i2<emf_->ny; i2++){
      j = find_index(gmg[0].n2+1, gmg[0].x2, emf_->x2[i2]);
      jp1 = MIN(j + 1, gmg[0].n2);
      w2 = (emf_->x2[i2] - gmg[0].x2[j]) / gmg[0].d2s[j];
      for(i1=0; i1<emf_->nx; i1++){
	i = find_index(gmg[0].n1, gmg[0].x1s, emf_->x1[i1]);
	ip1 = MIN(i + 1, gmg[0].n1-1);
	w1 = (emf_->x1[i1] - gmg[0].x1s[i]) / gmg[0].d1[ip1];
	
	s = 0;
	s += Ex[k][j][i] * (1. - w1) * (1. - w2) * (1. - w3);
	s += Ex[k][j][ip1] * w1 * (1. - w2) * (1. - w3);
	s += Ex[k][jp1][i] * (1. - w1) * w2 * (1. - w3);
	s += Ex[k][jp1][ip1] * w1 * w2 * (1. - w3);
	s += Ex[kp1][j][i] * (1. - w1) * (1. - w2) * w3;
	s += Ex[kp1][j][ip1] * w1 * (1. - w2) * w3;
	s += Ex[kp1][jp1][i] * (1. - w1) * w2 * w3;
	s += Ex[kp1][jp1][ip1] * w1 * w2 * w3;
	E[ipolar][ifreq][kk] = s;//extract Ex
	kk++;
      }
    }
  }
  for(i3=0; i3<emf_->nz; i3++){
    k = find_index(gmg[0].n3+1, gmg[0].x3, emf_->x3[i3]);
    kp1 = MIN(k + 1, gmg[0].n3);
    w3 = (emf_->x3[i3] - gmg[0].x3[k]) / gmg[0].d3s[k];
    for(i2=0; i2<emf_->ny; i2++){
      j = find_index(gmg[0].n2+1, gmg[0].x2s, emf_->x2[i2]);
      jp1 = MIN(j + 1, gmg[0].n2-1);
      w2 = (emf_->x2[i2] - gmg[0].x2s[j]) / gmg[0].d2[jp1];
      for(i1=0; i1<emf_->nx; i1++){
	i = find_index(gmg[0].n1, gmg[0].x1, emf_->x1[i1]);
	ip1 = MIN(i + 1, gmg[0].n1);
	w1 = (emf_->x1[i1] - gmg[0].x1[i]) / gmg[0].d1s[i];
	
	s = 0;
	s += Ey[k][j][i] * (1. - w1) * (1. - w2) * (1. - w3);
	s += Ey[k][j][ip1] * w1 * (1. - w2) * (1. - w3);
	s += Ey[k][jp1][i] * (1. - w1) * w2 * (1. - w3);
	s += Ey[k][jp1][ip1] * w1 * w2 * (1. - w3);
	s += Ey[kp1][j][i] * (1. - w1) * (1. - w2) * w3;
	s += Ey[kp1][j][ip1] * w1 * (1. - w2) * w3;
	s += Ey[kp1][jp1][i] * (1. - w1) * w2 * w3;
	s += Ey[kp1][jp1][ip1] * w1 * w2 * w3;
	E[ipolar][ifreq][kk] = s;//extract Ey
	kk++;
      }
    }
  }
  for(i3=0; i3<emf_->nz; i3++){
    double sigma_loc, sigma000, sigma001, sigma010, sigma011;
    double sigma100, sigma101, sigma110, sigma111;
    double w000, w001, w010, w011, w100, w101, w110, w111;
    int im1, ip1m1, jm1, jp1m1;

    k = find_index(gmg[0].n3+1, gmg[0].x3s, emf_->x3[i3]);
    kp1 = MIN(k + 1, gmg[0].n3-1);
    w3 = (emf_->x3[i3] - gmg[0].x3s[k]) / gmg[0].d3[k];
    for(i2=0; i2<emf_->ny; i2++){
      j = find_index(gmg[0].n2+1, gmg[0].x2, emf_->x2[i2]);
      jp1 = MIN(j + 1, gmg[0].n2);
      w2 = (emf_->x2[i2] - gmg[0].x2s[j]) / gmg[0].d2s[jp1];
      for(i1=0; i1<emf_->nx; i1++){
	i = find_index(gmg[0].n1, gmg[0].x1, emf_->x1[i1]);
	ip1 = MIN(i + 1, gmg[0].n1);
	w1 = (emf_->x1[i1] - gmg[0].x1[i]) / gmg[0].d1s[i];

	im1 = MAX(i - 1, 0);
	ip1m1 = MAX(ip1 - 1, 0);
	jm1 = MAX(j - 1, 0);
	jp1m1 = MAX(jp1 - 1, 0);

	w000 = (1. - w1) * (1. - w2) * (1. - w3);
	w001 = w1 * (1. - w2) * (1. - w3);
	w010 = (1. - w1) * w2 * (1. - w3);
	w011 = w1 * w2 * (1. - w3);
	w100 = (1. - w1) * (1. - w2) * w3;
	w101 = w1 * (1. - w2) * w3;
	w110 = (1. - w1) * w2 * w3;
	w111 = w1 * w2 * w3;

	sigma000 = 0.25*(gmg[0].sigma33[k][j][i] + gmg[0].sigma33[k][j][im1] + gmg[0].sigma33[k][jm1][i] + gmg[0].sigma33[k][jm1][im1]);
	sigma001 = 0.25*(gmg[0].sigma33[k][j][ip1] + gmg[0].sigma33[k][j][ip1m1] + gmg[0].sigma33[k][jm1][ip1] + gmg[0].sigma33[k][jm1][ip1m1]);
	sigma010 = 0.25*(gmg[0].sigma33[k][jp1][i] + gmg[0].sigma33[k][jp1][im1] + gmg[0].sigma33[k][jp1m1][i] + gmg[0].sigma33[k][jp1m1][im1]);
	sigma011 = 0.25*(gmg[0].sigma33[k][jp1][ip1] + gmg[0].sigma33[k][jp1][ip1m1] + gmg[0].sigma33[k][jp1m1][ip1] + gmg[0].sigma33[k][jp1m1][ip1m1]);
	sigma100 = 0.25*(gmg[0].sigma33[kp1][j][i] + gmg[0].sigma33[kp1][j][im1] + gmg[0].sigma33[kp1][jm1][i] + gmg[0].sigma33[kp1][jm1][im1]);
	sigma101 = 0.25*(gmg[0].sigma33[kp1][j][ip1] + gmg[0].sigma33[kp1][j][ip1m1] + gmg[0].sigma33[kp1][jm1][ip1] + gmg[0].sigma33[kp1][jm1][ip1m1]);
	sigma110 = 0.25*(gmg[0].sigma33[kp1][jp1][i] + gmg[0].sigma33[kp1][jp1][im1] + gmg[0].sigma33[kp1][jp1m1][i] + gmg[0].sigma33[kp1][jp1m1][im1]);
	sigma111 = 0.25*(gmg[0].sigma33[kp1][jp1][ip1] + gmg[0].sigma33[kp1][jp1][ip1m1] + gmg[0].sigma33[kp1][jp1m1][ip1] + gmg[0].sigma33[kp1][jp1m1][ip1m1]);

	s = 0;
	s += sigma000 * Ez[k][j][i] * w000;
	s += sigma001 * Ez[k][j][ip1] * w001;
	s += sigma010 * Ez[k][jp1][i] * w010;
	s += sigma011 * Ez[k][jp1][ip1] * w011;
	s += sigma100 * Ez[kp1][j][i] * w100;
	s += sigma101 * Ez[kp1][j][ip1] * w101;
	s += sigma110 * Ez[kp1][jp1][i] * w110;
	s += sigma111 * Ez[kp1][jp1][ip1] * w111;

	sigma_loc = sigma000 * w000 + sigma001 * w001 + sigma010 * w010 + sigma011 * w011
	  + sigma100 * w100 + sigma101 * w101 + sigma110 * w110 + sigma111 * w111;
	if (fabs(sigma_loc) < DBL_EPSILON) err("sigma33 interpolation is zero while extracting Ez at [%d,%d,%d]", i1, i2, i3);

	E[ipolar][ifreq][kk] = s / sigma_loc;//extract Jz then compute Ez=Jz/sigma
	kk++;
      }
    }
  }

}


void inject_adjoint_sources(acq_t *acq, int ifreq, int ipolar)
{
  int n1, n2, n3, n;
  int i, j, k, irec;
  int ip1, jp1, kp1;
  int im1, jm1, km1;
  double w1, w2, w3, vol;
  complex s;
  double *d1s, *d2s, *d3s;
  double ***invmur;
  complex hxp, hxm, hyp, hym, hzp, hzm;
  complex ***Ex, ***Ey, ***Ez;
  complex ***Hx, ***Hy, ***Hz;
  
  n = 3 * (gmg[0].n1 + 1) * (gmg[0].n2 + 1) * (gmg[0].n3 + 1);
  memset(&gmg[0].u[0][0][0][0], 0, n*sizeof(complex));
  memset(&gmg[0].r[0][0][0][0], 0, n*sizeof(complex));
  memset(&gmg[0].f[0][0][0][0], 0, n*sizeof(complex));
  Ex = gmg[0].u[0];
  Ey = gmg[0].u[1];
  Ez = gmg[0].u[2];
  Hx = gmg[0].r[0];
  Hy = gmg[0].r[1];
  Hz = gmg[0].r[2];
  d1s = gmg[0].d1s;
  d2s = gmg[0].d2s;
  d3s = gmg[0].d3s;
  invmur = gmg[0].invmur;
  
  //finally inject source terms for E and H
  for (irec = 0; irec < acq->nrec; irec++) {
    i = find_index(gmg[0].n1, gmg[0].x1s, acq->rec_x1[irec]);
    j = find_index(gmg[0].n2 + 1, gmg[0].x2, acq->rec_x2[irec]);
    k = find_index(gmg[0].n3 + 1, gmg[0].x3, acq->rec_x3[irec]);
    ip1 = MIN(i + 1, gmg[0].n1 - 1);
    jp1 = MIN(j + 1, gmg[0].n2);
    kp1 = MIN(k + 1, gmg[0].n3);
    w1 = (acq->rec_x1[irec] - gmg[0].x1s[i]) / gmg[0].d1[ip1];
    w2 = (acq->rec_x2[irec] - gmg[0].x2[j]) / gmg[0].d2s[j];
    w3 = (acq->rec_x3[irec] - gmg[0].x3[k]) / gmg[0].d3s[k];
    vol = gmg[0].d1[ip1]*gmg[0].d2s[j]*gmg[0].d3s[k];
    s = emf_->s_Ex[ipolar][ifreq][irec]/vol;//adjoint source density for Ex
    Ex[k][j][i] += s * (1. - w1) * (1. - w2) * (1. - w3);
    Ex[k][j][ip1] += s * w1 * (1. - w2) * (1. - w3);
    Ex[k][jp1][i] += s * (1. - w1) * w2 * (1. - w3);
    Ex[k][jp1][ip1] += s * w1 * w2 * (1. - w3);
    Ex[kp1][j][i] += s * (1. - w1) * (1. - w2) * w3;
    Ex[kp1][j][ip1] += s * w1 * (1. - w2) * w3;
    Ex[kp1][jp1][i] += s * (1. - w1) * w2 * w3;
    Ex[kp1][jp1][ip1] += s * w1 * w2 * w3;

    i = find_index(gmg[0].n1 + 1, gmg[0].x1, acq->rec_x1[irec]);
    j = find_index(gmg[0].n2, gmg[0].x2s, acq->rec_x2[irec]);
    k = find_index(gmg[0].n3 + 1, gmg[0].x3, acq->rec_x3[irec]);
    ip1 = MIN(i + 1, gmg[0].n1);
    jp1 = MIN(j + 1, gmg[0].n2 - 1);
    kp1 = MIN(k + 1, gmg[0].n3);
    w1 = (acq->rec_x1[irec] - gmg[0].x1[i]) / gmg[0].d1s[i];
    w2 = (acq->rec_x2[irec] - gmg[0].x2s[j]) / gmg[0].d2[jp1];
    w3 = (acq->rec_x3[irec] - gmg[0].x3[k]) / gmg[0].d3s[k];
    vol = gmg[0].d1s[i]*gmg[0].d2[jp1]*gmg[0].d3s[k];
    s = emf_->s_Ey[ipolar][ifreq][irec]/vol;//adjoint source density for Ey
    Ey[k][j][i] += s * (1. - w1) * (1. - w2) * (1. - w3);
    Ey[k][j][ip1] += s * w1 * (1. - w2) * (1. - w3);
    Ey[k][jp1][i] += s * (1. - w1) * w2 * (1. - w3);
    Ey[k][jp1][ip1] += s * w1 * w2 * (1. - w3);
    Ey[kp1][j][i] += s * (1. - w1) * (1. - w2) * w3;
    Ey[kp1][j][ip1] += s * w1 * (1. - w2) * w3;
    Ey[kp1][jp1][i] += s * (1. - w1) * w2 * w3;
    Ey[kp1][jp1][ip1] += s * w1 * w2 * w3;

    i = find_index(gmg[0].n1 + 1, gmg[0].x1, acq->rec_x1[irec]);
    j = find_index(gmg[0].n2, gmg[0].x2s, acq->rec_x2[irec]);
    k = find_index(gmg[0].n3, gmg[0].x3s, acq->rec_x3[irec]);
    ip1 = MIN(i + 1, gmg[0].n1);
    jp1 = MIN(j + 1, gmg[0].n2 - 1);
    kp1 = MIN(k + 1, gmg[0].n3);
    w1 = (acq->rec_x1[irec] - gmg[0].x1[i]) / gmg[0].d1s[i];
    w2 = (acq->rec_x2[irec] - gmg[0].x2s[j]) / gmg[0].d2[jp1];
    w3 = (acq->rec_x3[irec] - gmg[0].x3s[k]) / gmg[0].d3[kp1];
    vol = gmg[0].d1s[i]*gmg[0].d2[jp1]*gmg[0].d3[kp1];
    s = emf_->s_Hx[ipolar][ifreq][irec]/vol;//adjoint source density for Hx
    Hx[k][j][i] += s * (1. - w1) * (1. - w2) * (1. - w3);
    Hx[k][j][ip1] += s * w1 * (1. - w2) * (1. - w3);
    Hx[k][jp1][i] += s * (1. - w1) * w2 * (1. - w3);
    Hx[k][jp1][ip1] += s * w1 * w2 * (1. - w3);
    Hx[kp1][j][i] += s * (1. - w1) * (1. - w2) * w3;
    Hx[kp1][j][ip1] += s * w1 * (1. - w2) * w3;
    Hx[kp1][jp1][i] += s * (1. - w1) * w2 * w3;
    Hx[kp1][jp1][ip1] += s * w1 * w2 * w3;

    i = find_index(gmg[0].n1, gmg[0].x1s, acq->rec_x1[irec]);
    j = find_index(gmg[0].n2 + 1, gmg[0].x2, acq->rec_x2[irec]);
    k = find_index(gmg[0].n3, gmg[0].x3s, acq->rec_x3[irec]);
    ip1 = MIN(i + 1, gmg[0].n1 - 1);
    jp1 = MIN(j + 1, gmg[0].n2);
    kp1 = MIN(k + 1, gmg[0].n3 - 1);
    w1 = (acq->rec_x1[irec] - gmg[0].x1s[i]) / gmg[0].d1[ip1];
    w2 = (acq->rec_x2[irec] - gmg[0].x2[j]) / gmg[0].d2s[j];
    w3 = (acq->rec_x3[irec] - gmg[0].x3s[k]) / gmg[0].d3[kp1];
    vol = gmg[0].d1[ip1]*gmg[0].d2s[j]*gmg[0].d3[kp1];
    s = emf_->s_Hy[ipolar][ifreq][irec]/vol;//adjoint source density for Hy
    Hy[k][j][i] += s * (1. - w1) * (1. - w2) * (1. - w3);
    Hy[k][j][ip1] += s * w1 * (1. - w2) * (1. - w3);
    Hy[k][jp1][i] += s * (1. - w1) * w2 * (1. - w3);
    Hy[k][jp1][ip1] += s * w1 * w2 * (1. - w3);
    Hy[kp1][j][i] += s * (1. - w1) * (1. - w2) * w3;
    Hy[kp1][j][ip1] += s * w1 * (1. - w2) * w3;
    Hy[kp1][jp1][i] += s * (1. - w1) * w2 * w3;
    Hy[kp1][jp1][ip1] += s * w1 * w2 * w3;
  }

  //form the right hand side of adjoint equation
  n1 = gmg[0].n1;
  n2 = gmg[0].n2;
  n3 = gmg[0].n3;
  for(k=0; k<n3; k++){
    kp1 = MIN(k+1, n3);
    km1 = MAX(k-1, 0);
    for(j=0; j<n2; j++){
      jp1 = MIN(j+1, n2);
      jm1 = MAX(j-1, 0);
      for(i=0; i<n1; i++){
	ip1 = MIN(i+1, n1);
	im1 = MAX(i-1, 0);

	if(j>0 && k>0){
	  vol = gmg[0].d1s[i]*gmg[0].d2[j]*gmg[0].d3[k];//be aware of volume factor has been multiplied in invmur[][][]
	  hzp = Hz[k][j][i] * 0.5*(invmur[k][j][i] + invmur[km1][j][i]);//Hz(I,J,k)
	  hzm = Hz[k][jm1][i] * 0.5*(invmur[k][jm1][i] + invmur[km1][jm1][i]);//Hz(I,J-1,k)
	  hyp = Hy[k][j][i] * 0.5*(invmur[k][j][i] + invmur[k][jm1][i]);//Hy(I,j,K)
	  hym = Hy[km1][j][i] * 0.5*(invmur[km1][j][i] + invmur[km1][jm1][i]);//Hy(I,j,K-1)
	  gmg[0].f[0][k][j][i] = ((hzp/d2s[j]-hzm/d2s[jm1]) - (hyp/d3s[k]-hym/d3s[km1])) + I_omega_mu0*Ex[k][j][i]*vol;
	}

	if(i>0 && k>0){
	  vol = gmg[0].d1[i]*gmg[0].d2s[j]*gmg[0].d3[k];//be aware of volume factor has been multiplied in invmur[][][]
	  hxp = Hx[k][j][i] * 0.5*(invmur[k][j][i] + invmur[k][j][im1]);//Hx(i,J,K)
	  hxm = Hx[km1][j][i] * 0.5*(invmur[km1][j][i] + invmur[km1][j][im1]);//Hx(i,J,K-1)
	  hzp = Hz[k][j][i] * 0.5*(invmur[k][j][i] + invmur[km1][j][i]);//Hz(I,J,k)
	  hzm = Hz[k][j][im1] * 0.5*(invmur[k][j][im1] + invmur[km1][j][im1]);//Hz(I-1,J,k)
	  gmg[0].f[1][k][j][i] = ((hxp/d3s[k]-hxm/d3s[km1]) - (hzp/d1s[i]-hzm/d1s[im1])) + I_omega_mu0*Ey[k][j][i]*vol;
	}

	if(i>0 && j>0){
	  vol = gmg[0].d1[i]*gmg[0].d2[j]*gmg[0].d3s[k];//be aware of volume factor has been multiplied in invmur[][][]
	  hyp = Hy[k][j][i] * 0.5*(invmur[k][j][i] + invmur[k][jm1][i]);//Hy(I,j,K)
	  hym = Hy[k][j][im1] * 0.5*(invmur[k][j][im1] + invmur[k][jm1][im1]);//Hy(I-1,j,K)
	  hxp = Hx[k][j][i] * 0.5*(invmur[k][j][i] + invmur[k][j][im1]);//Hx(i,J,K)
	  hxm = Hx[k][jm1][i] * 0.5*(invmur[k][jm1][i] + invmur[k][jm1][im1]);//Hx(i,J-1,K)
	  gmg[0].f[2][k][j][i] = ((hyp/d1s[i]-hym/d1s[im1]) - (hxp/d2s[j]-hxm/d2s[jm1])) + I_omega_mu0*Ez[k][j][i]*vol;
	}
      }
    }
  }
  
}
