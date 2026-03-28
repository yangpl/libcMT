/* Binary-search helper for locating a coordinate interval.
 * Returns the largest index k such that x[k] <= val < x[k+1] in an
 * ascending grid array.
 *----------------------------------------------------------------------
 *   Copyright (c) Pengliang Yang
 *   E-mail: ypl.2100@gmail.com
 *   Homepage: https://yangpl.wordpress.com
 *----------------------------------------------------------------------
 */
int find_index(int n, double *x, double val)
{
  /*assume x[] has been sorted ascendingly */
  /* int i; */
  
  /* for(i=0; i<n; i++){ */
  /*   if(val<x[i]) break; */
  /* } */

  int low=0;
  int high = n-1;
  int i = (low+high)/2;
  
  while(low<high){
    i=(low+high)/2;
    if(x[i]<=val) low = i;
    if(x[i]>val) high=i;
    if(low==high||low==high-1) break;
  }

  return low;
}
