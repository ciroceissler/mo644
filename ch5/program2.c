//
// author      : ciro ceissler
// email       : ciro.ceissler at gmail.com
// description : trapezoidal rule program
//

#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#  include <omp.h>
#endif  //  _OPENMP

void trap(double a, double b, int n, double* global_result_p);
double local_trap(double a, double b, int n);
double parallel_trap(double a, double b, int n);

int main(int argc, char* argv[]) {

  int n;
  int thread_count = 1;

  double a;
  double b;
  double global_result = 0.0;

  printf("Enter a, b and n:\n");
  scanf("%lf %lf %d", &a, &b, &n);

  if (argc == 2) {
    thread_count = strtol(argv[1], NULL, 10);
  }

  // first version

# pragma omp parallel num_threads(thread_count)
  trap(a, b, n, &global_result);

  printf("Version 1: With n = %d trapezoids, our estimate\n", n);
  printf("of the integral from %f to %f = %.14e\n", a, b, global_result);

  // second version
  global_result = 0.0;

# pragma omp parallel num_threads(thread_count) reduction(+: global_result)
  global_result += local_trap(a, b, n);

  printf("\n");
  printf("Version 2: With n = %d trapezoids, our estimate\n", n);
  printf("of the integral from %f to %f = %.14e\n", a, b, global_result);

  // third version
  global_result = 0.0;

  global_result = parallel_trap(a, b, n);

  printf("\n");
  printf("Version 3: With n = %d trapezoids, our estimate\n", n);
  printf("of the integral from %f to %f = %.14e\n", a, b, global_result);

  return 0;
}  /* main */

double f(double x) {
  return x*x*x;
}  /* f */

void trap(double a, double b, int n, double* global_result_p) {

  double h;
  double x;
  double my_result;
  double local_a;
  double local_b;

  int i;
  int local_n;

#ifdef _OPENMP
  int my_rank      = omp_get_thread_num();
  int thread_count = omp_get_num_threads();
#else
  int my_rank      = 0;
  int thread_count = 1;
#endif  //  _OPENMP

  h = (b-a)/n;

  local_n = n/thread_count;
  local_a = a + my_rank*local_n*h;
  local_b = local_a + local_n*h;

  my_result = (f(local_a) + f(local_b))/2.0;

  for (i = 1; i <= local_n - 1; i++) {
    x = local_a + i*h;

    my_result += f(x);
  }

  my_result = my_result*h;

# pragma omp critical
  *global_result_p += my_result;

} /* trap */

double local_trap(double a, double b, int n) {

  double h;
  double x;
  double my_result;
  double local_a;
  double local_b;

  int i;
  int local_n;

#ifdef _OPENMP
  int my_rank      = omp_get_thread_num();
  int thread_count = omp_get_num_threads();
#else
  int my_rank      = 0;
  int thread_count = 1;
#endif  //  _OPENMP

  h = (b-a)/n;

  local_n = n/thread_count;
  local_a = a + my_rank*local_n*h;
  local_b = local_a + local_n*h;

  my_result = (f(local_a) + f(local_b))/2.0;

  for (i = 1; i <= local_n - 1; i++) {
    x = local_a + i*h;

    my_result += f(x);
  }

  my_result = my_result*h;

  return my_result;

} /* local_trap */

double parallel_trap(double a, double b, int n) {
  int i;
  double h;
  double approx;

#ifdef _OPENMP
  int my_rank      = omp_get_thread_num();
  int thread_count = omp_get_num_threads();
#else
  int my_rank      = 0;
  int thread_count = 1;
#endif  //  _OPENMP

  h = (b - a)/n;

  approx = (f(a) + f(b))/2.0;

# pragma omp parallel for num_threads(thread_count) reduction(+: approx)
  for (i = 1; i <= n - 1; i++) {
    approx += f(a + i*h);
  }

  approx = h*approx;

  return approx;
} /* parallel_trap */

// taf!
