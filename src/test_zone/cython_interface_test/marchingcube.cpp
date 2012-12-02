#include "marchingcube.hpp"
#include <iostream>

static double points[] = {
  1, 0.1,
  2, 0.2, 0.3,
  3, 0.4, 0.5, 0.6,
  0
};

int call_levelset(levelsetfunc user_func, void *user_data) {
  double *p = points;
  int d = 0;
  while (*p) {
    int dim = *(p++);
    d += user_func(p, dim, user_data);
    std::cout << "(C++) call_levelset: " << d << std::endl;
    p += dim;
  }
  return d;
}

int get_cells(double *array, int *m, int *n) {
  std::cout << "(C++) get_cells: " << *m << "x" << *n << std::endl;
  for(int i = 0; i < *m * *n; ++i)
    array[i] = i;
  int r = *m * *n;
  *m = 10;
  *n = 11;
  return r;
}
