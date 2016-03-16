//
// author      : ciro ceissler
// email       : ciro.ceissler at gmail.com
// description : bubble sort doesn't working
//

#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#  include <omp.h>
#endif  //  _OPENMP

void swap(int **list, int i) {

  int *a = *list;

  if (a[i] > a[i + 1]) {
    int tmp = a[i + 1];

    a[i + 1] = a[i];
    a[i] = tmp;
  }
} /* swap */

int main(int argc, char* argv[]) {

  int thread_count = 1;

  int list[10] = {7, 6, 5, 4, 3, 2, 1, 10, 8, 9};

  if (argc == 2) {
    thread_count = strtol(argv[1], NULL, 10);
  }

  seq_bubble_sort(list, 10);

# pragma omp parallel num_threads(thread_count)

  return 0;
}  /* main */

void seq_buuble_sort(int *list, int size) {

  int i;
  int list_length;

  for (list_length = size; list_length >= 2; list_length--) {
    for (i = 0; i < list_length - 1; i++) {
      swap(list, i);
    }
  }
}
