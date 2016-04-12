//
// name : Ciro Ceissler
// email: ciro.ceissler@gmail.com
// RA   : RA108786
//
// description: T4
//

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <sys/time.h>

// NOTES(ciroceissler): include pthread header
#include <pthread.h>

int rand_r(unsigned int *seedp);

// NOTES(ciroceissler): auxiliar macro used at 'parallel' monte_carlo_pi
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// NOTES(ciroceissler): global thread variables
int thread_count;
unsigned int n;

// NOTES(ciroceissler): divide bins calculation in a homonogeneous way among all threads
long step_bins;

/* long long unsigned int* in_array; */
long long unsigned int in_global = 0;

pthread_mutex_t lock;

/* long long unsigned int monte_carlo_pi(unsigned int n) { */
// NOTES(ciroceissler): parallel monte_carlo_pi
void* monte_carlo_pi(void* args) {
	long long unsigned int in = 0, i;
	double x, y, d;

  long thread;

  unsigned int seed = time(NULL);

  // NOTES(ciroceissler): take only thread number as an argument
  thread = (long) args;

  // NOTES(ciroceissler): calculate for specific range of bins, divide early among threads.
  // NOTES(ciroceissler): MIN macro avoid last thread calculate inexistent itens.
  for(i = thread*step_bins; i < MIN((thread*step_bins) + step_bins, n); i++) {
		x = ((rand_r(&seed) % 1000000)/500000.0)-1;
		y = ((rand_r(&seed) % 1000000)/500000.0)-1;
		d = ((x*x) + (y*y));
		if (d <= 1) {
      in+=1;
    }
	}

  pthread_mutex_lock(&lock);
  in_global += in;
  pthread_mutex_unlock(&lock);

  return NULL;
}

int main(void) {
	double pi;
	long unsigned int duracao;
	struct timeval start, end;

  long thread;

  pthread_t* thread_handles;

  // NOTES(ciroceissler): get number of threads
	scanf("%d %u",&thread_count, &n);

  // NOTES(ciroceissler): allocate storage to thread-specific information
  thread_handles = malloc (thread_count*sizeof(pthread_t));

  // NOTES(ciroceissler): divide number of bins among all threads.
  step_bins = ceil(n*1.0/thread_count*1.0);

	srand (time(NULL));

	gettimeofday(&start, NULL);

  // NOTES(ciroceissler): parallel function call with pthreads
  for (thread = 0; thread < thread_count; thread++) {
    pthread_create(&thread_handles[thread], NULL, monte_carlo_pi, (void*) thread);
  }

  // NOTES(ciroceissler): wait all threads complete
  for (thread = 0; thread < thread_count; thread++) {
    pthread_join(thread_handles[thread], NULL);
  }

	gettimeofday(&end, NULL);

	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));

	pi = 4*in_global/((double)n);
	printf("%lf\n%lu\n",pi,duracao);

  free(thread_handles);

	return 0;
}
