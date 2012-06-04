#include "utilities.h"

template< typename _T > _T power(const _T &x, const int &n) {
  if (n == 0) return 1;
  if (n == 1) return x;
  if (n & 1) return power(x, n-1) * x;
  _T tmp = power(x, n/2);
  return tmp * tmp;
}

double temperature(int k) {
  return T0 * power(T1, k);
}

double P(double old_e, double new_e, double T) {
  if (new_e < old_e) return 1.0;
  return 1 / (1 + exp((new_e - old_e) / T));
}

