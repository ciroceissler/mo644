//
// author      : ciro ceissler
// email       : ciro.ceissler at gmail.com
// description : fibonacci 
//

#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#  include <omp.h>
#endif  //  _OPENMP

void hello(void);

int main(int argc, char* argv[]) {

  int i;
  int n;
  int thread_count = 1;
  int *fibo;

  if (argc == 3) {
    thread_count = strtol(argv[1], NULL, 10);

    n = strtol(argv[2], NULL, 10);
  } else {
    thread_count = 1;
    n            = 2;
  }

  fibo = (int*) calloc(n, sizeof(int));

  fibo[0] = 1;
  fibo[1] = 1;

# pragma omp parallel for num_threads(thread_count)
  for (i = 2; i < n; i++) {
    fibo[i] = fibo[i-1] + fibo[i-2]; 
  }

  for (i = 0; i < n; i++) {
    printf("%d ", fibo[i]);
  }

  printf("\n");

  return 0;
}  /* main */

// taf!
