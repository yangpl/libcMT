/* Generates a nonuniform 1D grid by geometric stretching.
 * Solves for the geometric ratio that matches a target length while
 * preserving the requested first-cell size.
 *----------------------------------------------------------------------
 *   Copyright (c) Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *----------------------------------------------------------------------
 */
#include <math.h>

/*< create nonuniform grid using fixed point iteration >*/
double create_nugrid(int n, double len, double dx, double *x)
{
  int i;
  double q, qq;
  double eps = 1e-15;

  if(n*dx>=len) {
    for(i=0; i<=n; i++) x[i] = i*dx;
    return 1;
  }
  
  q = 1.1;
  qq = 1;
  while(1){
    qq = pow(len*(q-1.)/dx + 1., 1./n);
    if(fabs(qq-q)<eps) break;
    q = qq;
  }

  for(x[0]=0,i=1; i<=n; i++)
    x[i] = (pow(q,i) - 1.)*dx/(q-1.);

  return q;
}
