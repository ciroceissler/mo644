#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#include <omp.h>

// NOTES(ciroceissler): variaveis globas gerais
static char finalcmd[300] = "unzip -P%d -t %s 2>&1";
static char filename[100];
static int chunk_size = 10000;

//NOTES(ciroceissler) variavel global par ainformar que a senha foi encontrada
int has_finish;

FILE *popen(const char *command, const char *type);

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;

  stat = gettimeofday (&Tp, &Tzp);

  if (stat != 0) printf("Error return from gettimeofday: %d",stat);

  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

// NOTES(ciroceissler): metodo para computar a senha a partir de uma valor inicial
//                      (initial_i) ate o valor do chunk_size.
void test_passwd(int initial_i) {

  FILE * fp;
  char ret[200];
  char cmd[400];

  unsigned int k;

  // NOTES(ciroceissler): percorre todo o chunk_size
  for (k = initial_i; !has_finish && k < initial_i + chunk_size; k++) {

    sprintf((char*)&cmd, finalcmd, k, filename);

    fp = popen(cmd, "r");

    while (!feof(fp)) {
      fgets((char*)&ret, 200, fp);

      if (strcasestr(ret, "ok") != NULL) {
        printf("Senha:%d\n", k);

        // NOTES(ciroceissler): senha encontrada
        #pragma omp critical
        has_finish = 1;
      }

    }

    pclose(fp);
  }
}

int main () {
  int nt;

  double t_start, t_end;

  int i;

  scanf("%d", &nt);
  scanf("%s", filename);

  has_finish = 0;

  t_start = rtclock();

  #pragma omp parallel num_threads(nt) private(i) shared(chunk_size, has_finish)

  // NOTES(ciroceissler): thread de controle, inicia as outras threads e tambem
  //                      processa o workload.
  #pragma omp master
  {
    for(i=0; !has_finish && i < 500000; i += nt*chunk_size) {

      for(unsigned int j = 0; j < nt; j++) {
        #pragma omp task shared(finalcmd, filename)
        {
          test_passwd(i + j*chunk_size);
        }
      }
    }
  }

  t_end = rtclock();

  fprintf(stdout, "%0.6lf\n", t_end - t_start);
}

//
// RESULTADOS:
//
// == tempo de execucao paralelo:
//
// arq1.in:
// Senha:10000
// 0.006370
//
// arq2.in:
// Senha:100000
// 145.808934
//
// arq3.in:
// Senha:450000
// 530.587558
//
// arq4.in:
// Senha:310000
// 349.136113
//
// arq5.in:
// Senha:65000
// 50.764964
//
// arq6.in:
// Senha:245999
// 325.156417
//
// == tempo de execucao serial:
//
// arq1.in:
// Senha:10000
// 24.056035
//
// arq2.in:
// Senha:100000
// 240.915763
//
// arq3,in:
// Senha:450000
// 1079.184434
//
// arq4.in:
// Senha:310000
// 744.927912
//
// arq5.in:
// Senha:65000
// 155.354699
//
// arq6.in:
// Senha:245999
// 588.531458
//

// taf!
