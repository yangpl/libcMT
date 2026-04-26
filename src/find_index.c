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
  int low, high, mid;

  if(n < 2) return 0;

  if(val <= x[0]) return 0;
  if(val >= x[n-1]) return n-2;

  low = 0;
  high = n - 2;
  while(low <= high) {
    mid = (low + high) / 2;
    if(x[mid] <= val && val < x[mid + 1]) return mid;
    if(val < x[mid]) high = mid - 1;
    else low = mid + 1;
  }

  return n - 2;
}


int find_index_float(int n, float *x, double val)
{
  int low, high, mid;

  if(val <= x[0]) return 0;
  if(val >= x[n]) return n-1;

  low = 0;
  high = n-1;
  while(low <= high){
    mid = (low + high) / 2;
    if(x[mid] <= val && val < x[mid+1]) return mid;
    if(val < x[mid]) high = mid - 1;
    else low = mid + 1;
  }

  return n-1;
}
