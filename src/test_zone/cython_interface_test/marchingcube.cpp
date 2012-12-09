#include "marchingcube.hpp"
#include <iostream>


int get_cells(int dim, int *vertex_vals, int order, double *cell_data, int *m, double *facet_data, int *n){

  std::cout << "(C++) get_cells: dim=" << dim << "  order=" << order << "  m=" << *m << "  n=" << *n << std::endl;
  
  // your code here...
  for(int i = 0; i < *m; ++i)
    cell_data[i] = i;
  for(int i = 0; i < *n; ++i)
    facet_data[i] = i;
  *m = 10;
  *n = 7;

  // return success or fail
  return 0;
}
