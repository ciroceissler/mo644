//
// name : Ciro Ceissler
// email: ciro.ceissler@gmail.com
// RA   : RA108786
//
// description: T3
//

#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// NOTES(ciroceissler): include pthread header
#include <pthread.h>

// NOTES(ciroceissler): auxiliar macro used at 'parallel' count
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// NOTES(ciroceissler): global thread variables
int n;
int nval;
int *vet;
int thread_count;

double h;
double *val; 
double max;
double min;

// NOTES(ciroceissler): divide bins calculation in a homonogeneous way among all threads
long step_bins;

/* funcao que calcula o minimo valor em um vetor */
double min_val(double * vet,int nval) {
	int i;
	double min;

	min = FLT_MAX;

	for(i=0;i<nval;i++) {
		if(vet[i] < min)
			min =  vet[i];
	}

	return min;
}

/* funcao que calcula o maximo valor em um vetor */
double max_val(double * vet, int nval) {
	int i;
	double max;

	max = FLT_MIN;

	for(i=0;i<nval;i++) {
		if(vet[i] > max)
			max =  vet[i];
	}

	return max;
}

/* conta quantos valores no vetor estao entre o minimo e o maximo passados como parametros */
// NOTES(ciroceissler): parallel count
void* count(void* args) {
  int i; 
  int count;

  long j; 
  long thread;

  double min_t; 
  double max_t;

  // NOTES(ciroceissler): take only thread number as an argument
  thread = (long) args;

  // NOTES(ciroceissler): calculate for specific range of bins, divide early among threads.
  // NOTES(ciroceissler): MIN macro avoid last thread calculate inexistent itens.
  for(j = thread*step_bins; j < MIN((thread*step_bins) + step_bins, n); j++) {
    count = 0;
    min_t = min + j*h;
    max_t = min + (j+1)*h;

    for(i=0; i < nval; i++) {
      if(val[i] <= max_t && val[i] > min_t) {
        count++;
      }
    }

    vet[j] = count;
  }

  return NULL;
}

int main(int argc, char * argv[]) {
  int i;

	long unsigned int duracao;

	struct timeval start; 
  struct timeval end;

  long thread;

  pthread_t* thread_handles;

  // NOTES(ciroceissler): get number of threads
	scanf("%d",&thread_count);

  // NOTES(ciroceissler): allocate storage to thread-specific information
  thread_handles = malloc (thread_count*sizeof(pthread_t));

	/* entrada do numero de dados */
	scanf("%d",&nval);
	/* numero de barras do histograma a serem calculadas */
	scanf("%d",&n);

  // NOTES(ciroceissler): divide number of bins among all threads.
  step_bins = ceil(n*1.0/thread_count*1.0);

	/* vetor com os dados */
	val = (double *)malloc(nval*sizeof(double));
	vet = (int *)malloc(n*sizeof(int));

	/* entrada dos dados */
	for(i=0;i<nval;i++) {
		scanf("%lf",&val[i]);
	}

	/* calcula o minimo e o maximo valores inteiros */
	min = floor(min_val(val,nval));
	max = ceil(max_val(val,nval));

	/* calcula o tamanho de cada barra */
	h = (max - min)/n;

	gettimeofday(&start, NULL);

	/* chama a funcao */

  // NOTES(ciroceissler): parallel function call with pthreads
  for (thread = 0; thread < thread_count; thread++) {
    pthread_create(&thread_handles[thread], NULL, count, (void*) thread);
  }

  // NOTES(ciroceissler): wait all threads complete
  for (thread = 0; thread < thread_count; thread++) {
    pthread_join(thread_handles[thread], NULL);
  }

	gettimeofday(&end, NULL);

	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - \
	(start.tv_sec * 1000000 + start.tv_usec));

	printf("%.2lf",min);
	for(i=1;i<=n;i++) {
		printf(" %.2lf",min + h*i);
	}
	printf("\n");

	/* imprime o histograma calculado */
	printf("%d",vet[0]);
	for(i=1;i<n;i++) {
		printf(" %d",vet[i]);
	}
	printf("\n");

	/* imprime o tempo de duracao do calculo */
	printf("%lu\n",duracao);

	free(vet);
	free(val);
  free(thread_handles);

	return 0;
}
