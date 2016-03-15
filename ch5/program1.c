//
// author      : ciro ceissler
// email       : ciro.ceissler at gmail.com
// description : hello world
//

#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#  include <omp.h>
#endif  //  _OPENMP

void hello(void);

int main(int argc, char* argv[]) {

  int thread_count = 1;

  if (argc == 2) {
    thread_count = strtol(argv[1], NULL, 10);
  }

# pragma omp parallel num_threads(thread_count)
  hello();

  return 0;
}  /* main */

void hello(void) {
#ifdef _OPENMP
  int my_rank      = omp_get_thread_num();
  int thread_count = omp_get_num_threads();
#else
  int my_rank      = 0;
  int thread_count = 1;
#endif  //  _OPENMP

  printf("Hello from thread %d of %d\n", my_rank, thread_count);
}  /* hello */

// taf!
