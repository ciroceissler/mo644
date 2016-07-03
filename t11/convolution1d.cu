#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define Mask_Width 101

#define TILE_WIDTH 1000

__constant__ int M[Mask_Width];

__global__ void Convolution1D_kernel(int *N, int *P, int n) {
    
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j;

    int PValue = 0;

    int N_start_point = i - ((Mask_Width - 1)/2);

    for(j = 0; j < Mask_Width; j++) {
        if( ((N_start_point + j) >= 0) && ((N_start_point + j) < n) ) {
            PValue += N[N_start_point + j]*M[j];
        }
    }

    P[i] = PValue;
}

__global__ void Convolution1D_SM_kernel(int *N, int *P, int width) {

    // NOTES(ciroceissler): shared memory com tamanho suficiente para guardar os elementos
    //                      do halo da direita e esquerda, alem dos elementos centrais.
    __shared__ int N_ds[TILE_WIDTH + Mask_Width - 1];

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    int n = (Mask_Width - 1)/2;

    // NOTES(ciroceissler): calcula o indice do tile anterior
    int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;

    // NOTES(ciroceissler): carregar apenas o elementos necessarios do tile anterior
    if (threadIdx.x >= blockDim.x - n) {
        // NOTES(ciroceissler): checa se tem ou nao um elemento ghost, indice menor que zero.
        N_ds[threadIdx.x - (blockDim.x - n)] = (halo_index_left < 0) ? 0 : N[halo_index_left];
    }

    // NOTES(ciroceissler): carregar os elementos centrais
    N_ds[n + threadIdx.x] = N[blockIdx.x*blockDim.x + threadIdx.x];

    // NOTES(ciroceissler): calcula o indice do tile posterior
    int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;

    // NOTES(ciroceissler): carregar apenas o elementos necessarios do tile posterior
    if (threadIdx.x < n) {
        // NOTES(ciroceissler): checa se tem ou nao um elemento ghost, indice menor que zero.
        N_ds[n + blockDim.x + threadIdx.x] = (halo_index_right >= width) ? 0 : N[halo_index_right];
    }

    // NOTES(ciroceissler): sincronizar todas as threads
    __syncthreads();

    int PValue = 0;

    // NOTES(ciroceissler): calcular a operacao de convolucao
    for(int j = 0; j < Mask_Width; j++) {
        PValue += N_ds[threadIdx.x + j]*M[j];
    }

    P[i] = PValue;
}

#ifdef __DEBUG__
double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}
#endif // __DEBUG__

int main(int argc, char *argv[]) {

    // NOTES(ciroceissler): variaveis do device
    int *d_N, *d_P;

    /* input, output e máscara */ 
    int *N , *P, *h_M; 
    int n, i;

#ifdef __DEBUG__
    // NOTES(ciroceissler): variaveis de tempo
    double time_start;
    double time_end;
#endif // __DEBUG__

    /* Tamanho do vetor */	
    scanf("%d",&n);

    // NOTES(ciroceissler): dimensoes
    dim3 dimGrid(ceil((float)n / TILE_WIDTH), 1, 1);
    dim3 dimBlock(TILE_WIDTH, 1, 1);

    /* Alocação dos buffers necessários */
    P = (int *)malloc(n*sizeof(int));
    N = (int *)malloc(n*sizeof(int));
    h_M = (int *)malloc(sizeof(int)*Mask_Width);

    /* entrada dos valores */
    for(i = 0; i < n ; i++)
        scanf("%d",&N[i]);

    for(i = 0; i < Mask_Width; i++) h_M[i] = i;

#ifdef __DEBUG__
    time_start = rtclock();
#endif // __DEBUG__

    // NOTES(ciroceissler): alocacao dos buffers do device
    cudaMalloc((void **) &d_N, sizeof(int)*n);
    cudaMalloc((void **) &d_P, sizeof(int)*n);

    // NOTES(ciroceissler): copiar os vetores para o device
    cudaMemcpy(d_N, N, sizeof(int)*n, cudaMemcpyHostToDevice); 

    // NOTES(ciroceissler): copiar os valores para a constante
    cudaMemcpyToSymbol(M, h_M, sizeof(int)*Mask_Width);

    // NOTES(ciroceissler): rodar o kernel
    Convolution1D_SM_kernel<<<dimGrid, dimBlock>>>(d_N, d_P, n);

    // NOTES(ciroceissler): copiar o valor na GPU
    cudaMemcpy(P, d_P, sizeof(int)*n, cudaMemcpyDeviceToHost);

#ifdef __DEBUG__
    time_end = rtclock();
#endif // __DEBUG__

    for(i = 0; i < n; i++) printf("%d ", P[i]);
    printf("\n");

#ifdef __DEBUG__
    fprintf(stdout, "\n%0.6lf\n", time_end - time_start);
#endif // __DEBUG__

    cudaFree(d_N);
    cudaFree(d_P);

    free(P);
    free(N);
    free(h_M);
}

// ------------------------------------------------------------------------------------
// input   | cpu_serial  | gpu_nosharedmemory | gpu_sharedmemory | speedup (cpu/gpusm)
// ------------------------------------------------------------------------------------
// arq1.in | 0.057762    | 0.670367           | 0.621001         | 0.09301434297
// arq2.in | 0.578741    | 0.641411           | 0.636617         | 0.9090881959
// arq3.in | 5.779576    | 0.837708           | 0.820906         | 7.040484538
// ------------------------------------------------------------------------------------
// 
// taf!
