//
// author      : ciro ceissler
// email       : ciro.ceissler at gmail.com
// description : estimating pi
//

#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#  include <omp.h>
#endif  //  _OPENMP


int main(int argc, char* argv[]) {
  double factor = 1.0;
  double sum = 0.0;
  double result;

  int n = 1000000;

  int x = 5;

  int thread_count = 1;

  if (argc == 2) {
    thread_count = strtol(argv[1], NULL, 10);
  }

# pragma omp parallel for num_threads(thread_count) reduction(+: sum) private(factor)
  for (int k = 0; k < n; k++) {
    factor = (k%2 == 0) ? 1.0 : -1.0;

    sum += factor/(2*k + 1);
  }

  result = 4.0*sum;

  printf("sum = %f\n", result);

# pragma omp parallel num_threads(thread_count) default(none) private(x)
  {
    int my_rank = omp_get_thread_num();

    x = 1;

    printf("Thread %d > before initialization, x = %d\n", my_rank, x);

    x= my_rank;

    printf("Thread %d > after initialization, x = %d\n", my_rank, x);
  }

  printf("After parallel block, x = %d\n",x);

  return 0;
} /* main */
