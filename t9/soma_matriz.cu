#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 10

__global__ void add(int *a, int *b, int *c, int linhas, int colunas)
{
    int i = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int j = TILE_WIDTH * blockIdx.x + threadIdx.x;

    if (i < linhas & j < colunas)
    {
        c[i*colunas + j] = a[i*colunas + j] + b[i*colunas + j];
    }
}

int main()
{
    int *A, *B, *C;
    int *d_a, *d_b, *d_c;
    int i, j, n;

    //Input
    int linhas, colunas, size;

    scanf("%d", &linhas);
    scanf("%d", &colunas);

    n = linhas*colunas;
    size = sizeof(int)*n;

    //Alocando memória na CPU
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    //Inicializar
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            A[i*colunas+j] =  B[i*colunas+j] = i+j;
        }
    }

    // Copy inputs to device
    cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

    //Computacao que deverá ser movida para a GPU (que no momento é executada na CPU)
    //Lembrar que é necessário usar mapeamento 2D (visto em aula) 
    dim3 dimGrid(ceil((float)colunas / TILE_WIDTH), ceil((float)linhas / TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    add<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, linhas, colunas);

    // Copy result back to host
    cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

    long long int somador=0;
    //Manter esta computação na CPU
    for(i = 0; i < linhas; i++){
	    for(j = 0; j < colunas; j++){
            somador+=C[i*colunas+j];   
	    }
    }

    printf("%lli\n", somador);

    // Cleanup
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);

    free(A);
    free(B);
    free(C);
}

