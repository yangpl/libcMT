/* 1D MT boundary-condition solver.
 * Computes layered anisotropic MT responses and uses them to populate the
 * outer electric-field boundary values required by the 3D solve.
 *
 * This rewrite keeps the public interface unchanged, but re-implements the
 * layered 1D solver directly from the modal formulation in Section 4.2-4.3
 * of the attached note:
 *   - reduced symmetric conductivity tensor A = [[Axx, Axy], [Axy, Ayy]]
 *   - modal matrix M(z) from eq. (4.29)
 *   - layer transfer matrix S(h) = M(0) M(h)^(-1) from eq. (4.31)-(4.32)
 *   - decaying half-space boundary at the bottom
 *
 * With the unchanged function signature, sxx/syy/sxy are interpreted as the
 * reduced 2x2 coefficients Axx/Ayy/Axy after eliminating Ez. For the fully
 * general 3x3 conductivity tensor of eqs. (4.17)-(4.18), that reduction must
 * still be done by the caller before entering mt1d_solve_ani().
 *----------------------------------------------------------------------
 *   Copyright (c) Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *----------------------------------------------------------------------
 */
#include "cstd.h"
#include "gmg.h"

static int nlayer;
static double *h;

/* Return max |entry| over a 4x4 complex matrix. */
static double mat4_max_abs(const complex *A)
{
  int i;
  double amax = 0.0;
  for(i=0; i<16; ++i){
    double ai = cabs(A[i]);
    if(ai > amax) amax = ai;
  }
  return amax;
}

/* Multiply two 4x4 complex matrices stored in row-major order. */
static void mat4_mul(const complex *A, const complex *B, complex *C)
{
  int r, c, k;
  for(r=0; r<4; ++r){
    for(c=0; c<4; ++c){
      complex s = 0.0 + 0.0*I;
      for(k=0; k<4; ++k) s += A[4*r+k]*B[4*k+c];
      C[4*r+c] = s;
    }
  }
}

/* Multiply a 4x4 complex matrix by a 4-vector. */
static void mat4_vec_mul(const complex *A, const complex *x, complex *y)
{
  int r, k;
  for(r=0; r<4; ++r){
    complex s = 0.0 + 0.0*I;
    for(k=0; k<4; ++k) s += A[4*r+k]*x[k];
    y[r] = s;
  }
}

/* Solve the layered 1D anisotropic MT problem and return Ex/Ey at every layer node. */
void mt1d_solve_ani(double freq,
                    double *sxx, double *syy, double *sxy,
                    complex ExTop, complex EyTop,
                    complex *Ex, complex *Ey)
{
  int i, j;
  const int ncell = nlayer - 1;
  const int nlay = nlayer;
  const double MU0 = mu0;
  const double tiny = 1e-30;
  double omega = 2.0*PI*freq;
  complex iom = I*omega*MU0;
  double *Axx = NULL, *Ayy = NULL, *Axy = NULL;
  double *logSprod = NULL;
  complex *Sprod = NULL, *xip = NULL, *xim = NULL, *QpNeg = NULL, *QmNegRec = NULL;
  complex M[16], S[16], T[16], SM[16], Cbot[4], MC[4], F[4];
  complex G11, G12, G21, G22, detG, CD1, CD2;

  if(freq <= 0.0 || ncell < 1){
    for(i=0; i<nlay; ++i){
      Ex[i] = ExTop;
      Ey[i] = EyTop;
    }
    return;
  }

  Sprod = malloc((size_t)nlay * 16 * sizeof(complex));
  xip = malloc((size_t)nlay * sizeof(complex));
  xim = malloc((size_t)nlay * sizeof(complex));
  QpNeg = malloc((size_t)nlay * sizeof(complex));
  QmNegRec = malloc((size_t)nlay * sizeof(complex));
  logSprod = malloc((size_t)nlay * sizeof(double));
  Axx = malloc((size_t)nlay * sizeof(double));
  Ayy = malloc((size_t)nlay * sizeof(double));
  Axy = malloc((size_t)nlay * sizeof(double));
  if(Sprod == NULL || xip == NULL || xim == NULL || QpNeg == NULL || QmNegRec == NULL ||
     logSprod == NULL ||
     Axx == NULL || Ayy == NULL || Axy == NULL) goto fail;

  for(i=0; i<ncell; ++i){
    double axx = sxx[i];
    double ayy = syy[i];
    double axy = sxy[i];
    if(!isfinite(axx) || axx <= 0.0) axx = 1e-12;
    if(!isfinite(ayy) || ayy <= 0.0) ayy = 1e-12;
    if(!isfinite(axy)) axy = 0.0;
    Axx[i] = axx;
    Ayy[i] = ayy;
    Axy[i] = axy;
  }
  Axx[nlay-1] = Axx[nlay-2];
  Ayy[nlay-1] = Ayy[nlay-2];
  Axy[nlay-1] = Axy[nlay-2];

  memset(Sprod, 0, (size_t)nlay * 16 * sizeof(complex));
  for(i=0; i<4; ++i) Sprod[(nlay-1)*16 + 4*i + i] = 1.0 + 0.0*I;
  logSprod[nlay-1] = 0.0;
  memset(M, 0, sizeof(M));

  for(j=nlay-1; j>=0; --j){
    double ada = Axx[j] + Ayy[j];
    double add = Axx[j] - Ayy[j];
    double rad = add*add + 4.0*Axy[j]*Axy[j];
    double dA12 = (add >= 0.0) ? sqrt(rad) : -sqrt(rad);
    double A1 = 0.5*(ada + dA12);
    double A2 = 0.5*(ada - dA12);
    complex kp = csqrt(-iom) * csqrt(A1 + 0.0*I);
    complex km = csqrt(-iom) * csqrt(A2 + 0.0*I);
    complex qpn, qmr, qpq, dq, sp, sm, cp, cm;

    xip[j] = -kp / iom;
    xim[j] = -km / iom;

    if(fabs(Axy[j]) < tiny){
      qpn = 0.0 + 0.0*I;
      qmr = 0.0 + 0.0*I;
    }else{
      qpn = (2.0 * Axy[j]) / (add + dA12);
      qmr = 0.5 * (add - dA12) / Axy[j];
    }

    QpNeg[j] = qpn;
    QmNegRec[j] = qmr;
    qpq = qpn * qmr;
    dq = 1.0 - qpq;
    if(cabs(dq) < tiny) dq = tiny + 0.0*I;

    if(j == nlay-1){
      M[0]  = 0.0;            M[1]  = 1.0;             M[2]  = 0.0;            M[3]  = qmr;
      M[4]  = 0.0;            M[5]  = qpn;             M[6]  = 0.0;            M[7]  = 1.0;
      M[8]  = 0.0;            M[9]  = -xip[j] * qpn;   M[10] = 0.0;            M[11] = -xim[j];
      M[12] = 0.0;            M[13] = xip[j];          M[14] = 0.0;            M[15] = xim[j] * qmr;
      continue;
    }

    {
      complex aph = kp * h[j];
      complex amh = km * h[j];
      double layerShift = MAX(MAX(creal(aph), creal(amh)), 0.0);
      complex epp = cexp(aph - layerShift);
      complex epn = cexp(-aph - layerShift);
      complex emp = cexp(amh - layerShift);
      complex emn = cexp(-amh - layerShift);
      double logScale;
      double normScale;

      sp = 0.5 * (epp - epn);
      sm = 0.5 * (emp - emn);
      cp = 0.5 * (epp + epn);
      cm = 0.5 * (emp + emn);

      S[0]  =  cp - qpq * cm;
      S[1]  = -qmr * (cp - cm);
      S[2]  =  qmr * (sp / xip[j] - sm / xim[j]);
      S[3]  =  sp / xip[j] - qpq * sm / xim[j];

      S[4]  =  qpn * (cp - cm);
      S[5]  = -qpq * cp + cm;
      S[6]  =  qpq * sp / xip[j] - sm / xim[j];
      S[7]  =  qpn * (sp / xip[j] - sm / xim[j]);

      S[8]  = -qpn * (xip[j] * sp - xim[j] * sm);
      S[9]  =  qpq * xip[j] * sp - xim[j] * sm;
      S[10] = -qpq * cp + cm;
      S[11] = -qpn * (cp - cm);

      S[12] =  xip[j] * sp - qpq * xim[j] * sm;
      S[13] = -qmr * (xip[j] * sp - xim[j] * sm);
      S[14] =  qmr * (cp - cm);
      S[15] =  cp - qpq * cm;

      for(i=0; i<16; ++i) S[i] /= dq;
      mat4_mul(S, &Sprod[(j+1)*16], T);

      normScale = mat4_max_abs(T);
      if(!isfinite(normScale) || normScale < tiny) normScale = 1.0;
      for(i=0; i<16; ++i) Sprod[j*16 + i] = T[i] / normScale;

      logScale = logSprod[j+1] + layerShift + log(normScale);
      logSprod[j] = logScale;
    }
  }

  mat4_mul(&Sprod[0], M, SM);
  G11 = SM[1];
  G12 = SM[3];
  G21 = SM[5];
  G22 = SM[7];
  detG = G11 * G22 - G12 * G21;
  if(cabs(detG) < tiny) detG = tiny + 0.0*I;
  CD1 = ( G22 * ExTop - G12 * EyTop) / detG;
  CD2 = (-G21 * ExTop + G11 * EyTop) / detG;

  Cbot[0] = 0.0 + 0.0*I;
  Cbot[1] = CD1;
  Cbot[2] = 0.0 + 0.0*I;
  Cbot[3] = CD2;
  mat4_vec_mul(M, Cbot, MC);

  for(j=0; j<nlay; ++j){
    double ratio = exp(logSprod[j] - logSprod[0]);
    mat4_vec_mul(&Sprod[j*16], MC, F);
    Ex[j] = ratio * F[0];
    Ey[j] = ratio * F[1];
    if(!isfinite(creal(Ex[j])) || !isfinite(cimag(Ex[j]))) Ex[j] = 0.0 + 0.0*I;
    if(!isfinite(creal(Ey[j])) || !isfinite(cimag(Ey[j]))) Ey[j] = 0.0 + 0.0*I;
  }

  free(Sprod);
  free(xip);
  free(xim);
  free(QpNeg);
  free(QmNegRec);
  free(logSprod);
  free(Axx);
  free(Ayy);
  free(Axy);
  return;

fail:
  for(j=0; j<nlay; ++j){
    Ex[j] = ExTop;
    Ey[j] = EyTop;
  }
  free(Sprod);
  free(xip);
  free(xim);
  free(QpNeg);
  free(QmNegRec);
  free(logSprod);
  free(Axx);
  free(Ayy);
  free(Axy);
}

/* Cache layer thicknesses derived from the vertical node coordinates. */
void mt1d_init(int nlayer_, double *z)
{
  int i;
  nlayer = nlayer_;
  h = malloc((nlayer-1)*sizeof(double));
  for(i=0; i<nlayer-1; ++i) h[i] = z[i+1] - z[i];
}

/* Release the cached 1D layer-thickness array. */
void mt1d_free(void)
{
  free(h);
}

                    
/* Sample laterally averaged 1D conductivities and inject the resulting fields on the 3D boundary. */
void mt1d_efield_at_boundary(gmg_t *gmg, double freq, int ipolar)
{
  int i, j, k;
  int im1, iu, jm1, km1;
  int n1, n2, n3, n;
  int ju;
  complex ***Ex, ***Ey;
  double ***sigma11, ***sigma22;
  double *d1s, *d2s, *d3s;
  double *sxx1d, *syy1d, *sxy1d;
  double a1, a2, a3, a4, asum, sig;
  complex *Ex1d, *Ey1d;

  n1 = gmg[0].n1;
  n2 = gmg[0].n2;
  n3 = gmg[0].n3;
  n = 3 * (n1 + 1) * (n2 + 1) * (n3 + 1);
  memset(&gmg[0].u[0][0][0][0], 0, n * sizeof(complex));

  Ex = gmg[0].u[0];
  Ey = gmg[0].u[1];
  d1s = gmg[0].d1s;
  d2s = gmg[0].d2s;
  d3s = gmg[0].d3s;
  sigma11 = emf_->sigma11;
  sigma22 = emf_->sigma22;

  sxx1d = malloc(n3 * sizeof(double));
  syy1d = malloc(n3 * sizeof(double));
  sxy1d = malloc(n3 * sizeof(double));
  Ex1d = malloc((n3 + 1) * sizeof(complex));
  Ey1d = malloc((n3 + 1) * sizeof(complex));
  mt1d_init(n3 + 1, gmg[0].x3);

  /* XY polarization: impose Ex-driven boundary fields along the x-directed edges. */
  if (ipolar == 0) {
    for (j = 0; j <= n2; j++) {
      for (i = 0; i < n1; i++) {
        /* Average the surrounding 3D conductivities into a local 1D column. */
        for (k = 0; k < n3; k++) {
          jm1 = MAX(j - 1, 0);
          ju = MIN(j, n2 - 1);
          km1 = MAX(k - 1, 0);
          a1 = d2s[ju] * d3s[k];
          a2 = d2s[jm1] * d3s[k];
          a3 = d2s[ju] * d3s[km1];
          a4 = d2s[jm1] * d3s[km1];
          asum = a1 + a2 + a3 + a4;
          sig = (sigma11[k][ju][i] * a1
                 + sigma11[k][jm1][i] * a2
                 + sigma11[km1][ju][i] * a3
                 + sigma11[km1][jm1][i] * a4) / asum;
          sxx1d[k] = (isfinite(sig) && sig > 0.0) ? sig : 1e-12;

          sig = (sigma22[k][ju][i] * a1
                 + sigma22[k][jm1][i] * a2
                 + sigma22[km1][ju][i] * a3
                 + sigma22[km1][jm1][i] * a4) / asum;
          syy1d[k] = (isfinite(sig) && sig > 0.0) ? sig : 1e-12;
          sxy1d[k] = 0.0;
        }
        mt1d_solve_ani(freq, sxx1d, syy1d, sxy1d, 1.0 + 0.0 * I, 0.0 + 0.0 * I, Ex1d, Ey1d);
        /* Always set the top and bottom edges; side walls receive the full profile. */
        Ex[0][j][i] = Ex1d[0];
        Ey[0][j][i] = Ey1d[0];
        Ex[n3][j][i] = Ex1d[n3];
        Ey[n3][j][i] = Ey1d[n3];
        if (j == 0 || j == n2) {
          for (k = 0; k <= n3; k++) {
            Ex[k][j][i] = Ex1d[k];
            Ey[k][j][i] = Ey1d[k];
          }
        }
      }
    }
  /* YX polarization: impose Ey-driven boundary fields along the y-directed edges. */
  } else if (ipolar == 1) {
    for (j = 0; j < n2; j++) {
      for (i = 0; i <= n1; i++) {
        /* Average the surrounding 3D conductivities into a local 1D column. */
        for (k = 0; k < n3; k++) {
          im1 = MAX(i - 1, 0);
          iu = MIN(i, n1 - 1);
          km1 = MAX(k - 1, 0);
          a1 = d1s[iu] * d3s[k];
          a2 = d1s[im1] * d3s[k];
          a3 = d1s[iu] * d3s[km1];
          a4 = d1s[im1] * d3s[km1];
          asum = a1 + a2 + a3 + a4;
          sig = (sigma11[k][j][iu] * a1
                 + sigma11[k][j][im1] * a2
                 + sigma11[km1][j][iu] * a3
                 + sigma11[km1][j][im1] * a4) / asum;
          sxx1d[k] = (isfinite(sig) && sig > 0.0) ? sig : 1e-12;

          sig = (sigma22[k][j][iu] * a1
                 + sigma22[k][j][im1] * a2
                 + sigma22[km1][j][iu] * a3
                 + sigma22[km1][j][im1] * a4) / asum;
          syy1d[k] = (isfinite(sig) && sig > 0.0) ? sig : 1e-12;
          sxy1d[k] = 0.0;
        }
        mt1d_solve_ani(freq, sxx1d, syy1d, sxy1d, 0.0 + 0.0 * I, 1.0 + 0.0 * I, Ex1d, Ey1d);
        /* Always set the top and bottom edges; side walls receive the full profile. */
        Ex[0][j][i] = Ex1d[0];
        Ey[0][j][i] = Ey1d[0];
        Ex[n3][j][i] = Ex1d[n3];
        Ey[n3][j][i] = Ey1d[n3];
        if (i == 0 || i == n1) {
          for (k = 0; k <= n3; k++) {
            Ex[k][j][i] = Ex1d[k];
            Ey[k][j][i] = Ey1d[k];
          }
        }
      }
    }
  }
  mt1d_free();
  free(sxx1d);
  free(syy1d);
  free(sxy1d);
  free(Ex1d);
  free(Ey1d);
}
