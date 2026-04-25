/* Builds the padded computational mesh used by the multigrid solver.
 * Extends the input resistivity model with frequency-dependent padding
 * cells and maps source-model properties onto the computational grid.
 *----------------------------------------------------------------------
 *   Copyright (c) Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *----------------------------------------------------------------------
 */
#include "cstd.h"
#include "acq.h"
#include "emf.h"
 
int find_index(int n, double *x, double val);
int find_index_float(int n, float *x, double val);
double create_nugrid(int n, double len, double dx, double *x);

static int get_padding_count(double target_dist, double dx0, int min_cells, double qmax)
{
  int count = 0;
  double dist = 0.0;
  double dx = dx0;

  while(count < min_cells || dist < target_dist){
    dist += dx;
    dx *= qmax;
    count++;
  }

  return count;
}

int find_good_size(int n)
{
  int m, p, nn;

  m = 1;
  while(m<n) m *= 2;/* First candidate is the next power of two. */

  nn = m;  
  /* p = 7*m/8; */
  /* if(p>n && p<nn) nn = p; */
  p = 3*m/4;
  if(p>n && p<nn) nn = p;
  p = 5*m/8;
  if(p>n && p<nn) nn = p;
    
  return nn;
}

static void build_axis_with_padding(
  int nsrc,
  float *src,
  double left_dist,
  double right_dist,
  int min_pad,
  double qmax,
  int *nout,
  double **xout)
{
  int i;
  int nleft, nright, ntotal, nextra;
  double dx_left, dx_right;
  double *x, *pad;

  dx_left = src[1] - src[0];
  dx_right = src[nsrc] - src[nsrc-1];
  nleft = get_padding_count(left_dist, dx_left, min_pad, qmax);
  nright = get_padding_count(right_dist, dx_right, min_pad, qmax);
  ntotal = nsrc + nleft + nright;
  /* Add shell cells until the total size is friendly to the multigrid hierarchy. */
  nextra = find_good_size(ntotal) - ntotal;
  nleft += nextra/2;
  nright += nextra - nextra/2;
  
  *nout = nsrc + nleft + nright;
  x = alloc1double((*nout) + 1);

  for(i=0; i<=nsrc; i++) x[nleft + i] = src[i];

  x[nleft] = src[0];
  pad = alloc1double(nleft + 1);
  create_nugrid(nleft, left_dist, dx_left, pad);
  for(i=1; i<=nleft; i++){
    x[nleft-i] = src[0] - pad[i];
  }
  free1double(pad);

  x[nleft + nsrc] = src[nsrc];
  pad = alloc1double(nright + 1);
  create_nugrid(nright, right_dist, dx_right, pad);
  for(i=1; i<=nright; i++){
    x[nleft + nsrc + i] = src[nsrc] + pad[i];
  }
  free1double(pad);

  *xout = x;
}

static void set_air_cell(emf_t *emf, int i1, int i2, int i3, double rho_air)
{
  double sigma_air = 1.0/rho_air;

  emf->sigma11[i3][i2][i1] = sigma_air;
  emf->sigma22[i3][i2][i1] = sigma_air;
  emf->sigma33[i3][i2][i1] = sigma_air;
  emf->invmur[i3][i2][i1] = 1.0;
}

static void set_source_cell(emf_t *emf, int i1, int i2, int i3, int ix, int iy, int iz)
{
  emf->sigma11[i3][i2][i1] = 1.0/emf->rho11[iz][iy][ix];
  emf->sigma22[i3][i2][i1] = 1.0/emf->rho22[iz][iy][ix];
  emf->sigma33[i3][i2][i1] = 1.0/emf->rho33[iz][iy][ix];
  emf->invmur[i3][i2][i1] = 1.0;
}

/* Build the padded computational grid and map the input resistivity model onto it. */
void extend_model_init(emf_t *emf, int ifreq)
{
  int i1, i2, i3;
  int nb;
  double lextend, skin_depth, rho_skin;
  double x1c, x2c, x3c;
  int ix, iy, iz;
  double rho_air;
  
  if(emf->verb) printf("------- extend_model_init --------\n");
  if(!getparint("nb", &nb)) nb = 10;
  if(!getpardouble("rho_skin", &rho_skin)) rho_skin = emf->rhomax_noair;

  skin_depth = sqrt(2.0 * rho_skin / (2.*PI*emf->freqs[ifreq] * mu0));
  lextend = MIN(200e3,MAX(50e3, 4*skin_depth));/* Keep padding between 50 km and 200 km. */

  build_axis_with_padding(emf->nx, emf->x1node, lextend, lextend, nb, 1.3, &emf->n1, &emf->x1);/* Horizontal padding grows gently. */
  build_axis_with_padding(emf->ny, emf->x2node, lextend, lextend, nb, 1.3, &emf->n2, &emf->x2);/* Horizontal padding grows gently. */
  build_axis_with_padding(emf->nz, emf->x3node, lextend, lextend, nb, 2.0, &emf->n3, &emf->x3);/* Vertical padding may grow faster. */

  emf->sigma11 = alloc3double(emf->n1, emf->n2, emf->n3);
  emf->sigma22 = alloc3double(emf->n1, emf->n2, emf->n3);
  emf->sigma33 = alloc3double(emf->n1, emf->n2, emf->n3);
  emf->invmur = alloc3double(emf->n1, emf->n2, emf->n3);
  rho_air = MAX(emf->rho_air, emf->rhomax);

  for(i3=0; i3<emf->n3; i3++){
    x3c = 0.5*(emf->x3[i3] + emf->x3[i3+1]);
    for(i2=0; i2<emf->n2; i2++){
      x2c = 0.5*(emf->x2[i2] + emf->x2[i2+1]);
      for(i1=0; i1<emf->n1; i1++){
        x1c = 0.5*(emf->x1[i1] + emf->x1[i1+1]);

        if(x3c < emf->x3node[0]){/* Cells above the input model are always air. */
          set_air_cell(emf, i1, i2, i3, rho_air);
          continue;
        }

        ix = find_index_float(emf->nx, emf->x1node, x1c);
        iy = find_index_float(emf->ny, emf->x2node, x2c);
        iz = find_index_float(emf->nz, emf->x3node, x3c);
        set_source_cell(emf, i1, i2, i3, ix, iy, iz);
      }
    }
  }

  if(emf->verb){
    printf("rho_skin=%g Ohm m, skin_depth=%g, lextend=%g m\n", rho_skin, skin_depth, lextend);
    printf("computational mesh [n1, n2, n3]=[%d, %d, %d]\n", emf->n1, emf->n2, emf->n3);
    printf("computational domain [x1min, x1max]=[%g, %g]\n", emf->x1[0], emf->x1[emf->n1]);
    printf("computational domain [x2min, x2max]=[%g, %g]\n", emf->x2[0], emf->x2[emf->n2]);
    printf("computational domain [x3min, x3max]=[%g, %g]\n", emf->x3[0], emf->x3[emf->n3]);
  }
}

/*< free variables in emf >*/
void extend_model_free(emf_t *emf)
{
  free1double(emf->x1);
  free1double(emf->x2);
  free1double(emf->x3);
  free3double(emf->sigma11);
  free3double(emf->sigma22);
  free3double(emf->sigma33);
  free3double(emf->invmur);
}
