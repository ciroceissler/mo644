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

// NOTES(ciroceissler): pthread header
#include <pthread.h>

int rand_r(unsigned int *seedp);

// NOTES(ciroceissler): auxiliar macro used at 'parallel' monte_carlo_pi
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// NOTES(ciroceissler): variaveis globais para thread
int thread_count;
unsigned int n;
long long unsigned int in_global = 0;

// NOTES(ciroceissler): divide de uma maneira homogenea entre as threads
long step_bins;

// NOTES(ciroceissler): lock para in_global nao ter concorrencia
pthread_mutex_t lock;

// NOTES(ciroceissler): parallel monte_carlo_pi
void* monte_carlo_pi(void* args) {
	long long unsigned int in = 0, i;
	double x, y, d;

  long thread;

  unsigned int seed = time(NULL);

  // NOTES(ciroceissler): pegue o numero da thread
  thread = (long) args;

  // NOTES(ciroceissler): apenas um range especifico
  // NOTES(ciroceissler): MIN macro evita a ultima thread calcular passos inexistentes
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

  // NOTES(ciroceissler): pegar o numero de threads
	scanf("%d %u",&thread_count, &n);

  // NOTES(ciroceissler): allocate storage to thread-specific information
  thread_handles = malloc (thread_count*sizeof(pthread_t));

  // NOTES(ciroceissler): divide as atividades entre as threads
  step_bins = ceil(n*1.0/thread_count*1.0);

	srand (time(NULL));

	gettimeofday(&start, NULL);

  // NOTES(ciroceissler): chamada da funcao com pthreads
  for (thread = 0; thread < thread_count; thread++) {
    pthread_create(&thread_handles[thread], NULL, monte_carlo_pi, (void*) thread);
  }

  // NOTES(ciroceissler): espere todas as threads terminarem
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
