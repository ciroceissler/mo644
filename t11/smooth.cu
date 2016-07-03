#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define MASK_WIDTH 5
#define BLOCK_SIZE 8

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

typedef struct {
    unsigned char red, green, blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


static PPMImage *readPPM(const char *filename) {
    char buff[16];
    PPMImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }

    img = (PPMImage *) malloc(sizeof(PPMImage));
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n')
            ;
        c = getc(fp);
    }

    ungetc(c, fp);
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        exit(1);
    }

    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
                filename);
        exit(1);
    }

    if (rgb_comp_color != RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        exit(1);
    }

    while (fgetc(fp) != '\n')
        ;
    img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(PPMImage *img) {

    fprintf(stdout, "P6\n");
    fprintf(stdout, "# %s\n", COMMENT);
    fprintf(stdout, "%d %d\n", img->x, img->y);
    fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

    fwrite(img->data, 3 * img->x, img->y, stdout);

    fclose(stdout);
}

__global__ void smoothing_kernel(PPMPixel *image, PPMPixel *image_copy, int size_x, int size_y) {
    __shared__ PPMPixel image_ds[BLOCK_SIZE + MASK_WIDTH - 1][BLOCK_SIZE + MASK_WIDTH - 1];

    int total_red; 
    int total_blue; 
    int total_green;

    int n = (MASK_WIDTH - 1)/2;

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    // NOTES(ciroceissler): calcula apenas o elementos internos
    image_ds[n + threadIdx.x][n + threadIdx.y]  = image_copy[(y * size_x) + x] ;

    int halo_index_left   = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
    int halo_index_right  = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
    int halo_index_top    = (blockIdx.y - 1)*blockDim.y + threadIdx.y;
    int halo_index_bottom = (blockIdx.y + 1)*blockDim.y + threadIdx.y;

    // NOTES(ciroceissler): calcula os elementos do halo left
    if (threadIdx.x >= blockDim.x - n) {
        if (halo_index_left >= 0) {
            image_ds[threadIdx.x - (blockDim.x - n)][threadIdx.y + n]
                = image_copy[(y * size_x) + halo_index_left];
        } else {
            image_ds[threadIdx.x - (blockDim.x - n)][threadIdx.y + n] = (PPMPixel) {0,0,0};
        }
    }

    // NOTES(ciroceissler): calcula os elementos do halo right
    if (threadIdx.x < n) {
        if (halo_index_right < size_x) {
            image_ds[n + blockDim.x + threadIdx.x][threadIdx.y + n]
                = image_copy[(y * size_x) + halo_index_right];
        } else {
            image_ds[n + blockDim.x + threadIdx.x][threadIdx.y + n] = (PPMPixel) {0,0,0};
        }
    }

    // NOTES(ciroceissler): calcula os elementos do halo top
    if (threadIdx.y >= blockDim.y - n) {
        if (halo_index_top >= 0) {
            image_ds[threadIdx.x + n][threadIdx.y - (blockDim.y - n)]
                = image_copy[(halo_index_top * size_x) + x];
        } else {
            image_ds[threadIdx.x + n][threadIdx.y - (blockDim.y - n)] = (PPMPixel) {0,0,0};
        }
    }

    // NOTES(ciroceissler): calcula os elementos do halo bottom
    if (threadIdx.y < n) {
        if (halo_index_bottom < size_y) {
            image_ds[threadIdx.x + n][n + blockDim.y + threadIdx.y]
                = image_copy[(halo_index_bottom * size_x) + x];
        } else {
            image_ds[threadIdx.x + n][n + blockDim.y + threadIdx.y] = (PPMPixel) {0,0,0};
        }
    }

    // NOTES(ciroceissler): calcula os elementos do halo top-left
    if (threadIdx.x >= blockDim.x - n && threadIdx.y >= blockDim.y - n) {
        if (halo_index_left >= 0 && halo_index_top >= 0) {
            image_ds[threadIdx.x - (blockDim.x - n)][threadIdx.y - (blockDim.y - n)]
                = image_copy[(halo_index_top * size_x) + halo_index_left];
        } else {
            image_ds[threadIdx.x - (blockDim.x - n)][threadIdx.y - (blockDim.y - n)] 
                = (PPMPixel) {0,0,0};
        }
    }

    // NOTES(ciroceissler): calcula os elementos do halo top-right
    if (threadIdx.x < n && threadIdx.y >= blockDim.y - n) {
        if (halo_index_right < size_x && halo_index_top >= 0) {
            image_ds[n + blockDim.x + threadIdx.x][threadIdx.y - (blockDim.y - n)]
                = image_copy[(halo_index_top * size_x) + halo_index_right];
        } else {
            image_ds[n + blockDim.x + threadIdx.x][threadIdx.y - (blockDim.y - n)]
                = (PPMPixel) {0,0,0};
        }
    }

    // NOTES(ciroceissler): calcula os elementos do halo bottom-left
    if (threadIdx.x >= blockDim.x - n && threadIdx.y < n) {
        if (halo_index_left >= 0 && halo_index_bottom < size_y) {
            image_ds[threadIdx.x - (blockDim.x - n)][n + blockDim.y + threadIdx.y]
                = image_copy[(halo_index_bottom * size_x) + halo_index_left];
        } else {
            image_ds[threadIdx.x - (blockDim.x - n)][n + blockDim.y + threadIdx.y]
                = (PPMPixel) {0,0,0};
        }
    }
    
    // NOTES(ciroceissler): calcula os elementos do halo bottom-right
    if (threadIdx.x < n && threadIdx.y < n) {
        if (halo_index_right < size_x && halo_index_bottom < size_y) {
            image_ds[n + blockDim.x + threadIdx.x][n + blockDim.y + threadIdx.y]
                = image_copy[(halo_index_bottom * size_x) + halo_index_right];
        } else {
            image_ds[n + blockDim.x + threadIdx.x][n + blockDim.y + threadIdx.y]
                = (PPMPixel) {0,0,0};
        }
    }

    // NOTES(ciroceissler): sincronizar todas as threads
    __syncthreads();

    total_red   = 0;
    total_blue  = 0;
    total_green = 0;

    // NOTES(ciroceissler): calculo do smooth
    for (int j = threadIdx.y; j < threadIdx.y + MASK_WIDTH; j++) {
        for (int i = threadIdx.x; i < threadIdx.x + MASK_WIDTH; i++) {
            total_red   += image_ds[i][j].red   ;
            total_blue  += image_ds[i][j].blue  ;
            total_green += image_ds[i][j].green ;
        } 
    } 

    image[(y * size_x) + x].red   = total_red   / (MASK_WIDTH*MASK_WIDTH);
    image[(y * size_x) + x].blue  = total_blue  / (MASK_WIDTH*MASK_WIDTH);
    image[(y * size_x) + x].green = total_green / (MASK_WIDTH*MASK_WIDTH);
}

int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }

#ifdef __DEBUG__
    double t_start, t_end;
#endif // __DEBUG__

    char *filename = argv[1]; //Recebendo o arquivo!;

    std::size_t size_image;

    PPMImage *image = readPPM(filename);
    PPMImage *image_output = readPPM(filename);

    size_image = sizeof(PPMPixel)*image->x*image->y;

    // NOTES(ciroceissler): variaveis do device
    PPMPixel *d_image;
    PPMPixel *d_image_output;

    // NOTES(ciroceissler): dimensoes
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(ceil((float)image->x / BLOCK_SIZE), ceil((float)image->y / BLOCK_SIZE), 1);

#ifdef __DEBUG__
    t_start = rtclock();
#endif // __DEBUG__

    // NOTES(ciroceissler): alocacao dos buffers do device
    cudaMalloc((void **) &d_image       , size_image);
    cudaMalloc((void **) &d_image_output, size_image);

    // NOTES(ciroceissler): copiar os vetores para o device
    cudaMemcpy(d_image, image->data, size_image, cudaMemcpyHostToDevice); 

    // NOTES(ciroceissler): execucaco do kernel
    smoothing_kernel<<<dimGrid, dimBlock>>>(d_image_output, d_image, image->x, image->y);
    cudaDeviceSynchronize();

    // NOTES(ciroceissler): copiar o valor da GPU
    cudaMemcpy(image_output->data, d_image_output, size_image, cudaMemcpyDeviceToHost);


#ifdef __DEBUG__
    t_end = rtclock();

    fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);  
#endif // __DEBUG__

    writePPM(image_output);

    free(image);
    free(image_output);
}

// ------------------------------------------------------------------------------------
// input    | cpu_serial  | gpu_nosharedmemory | gpu_sharedmemory | speedup (cpu/gpusm)
// ------------------------------------------------------------------------------------
// arq1.ppm | 0.302824    | 0.667255           | 0.633555         |  0.4779758663
// arq2.ppm | 0.678099    | 0.755541           | 0.655261         |  1.034853287
// arq3.ppm | 2.710807    | 1.076326           | 0.751093         |  3.609149599
// ------------------------------------------------------------------------------------
// 
// - numero de acesso a memoria global substituido por shared memory:
//
// TILE_ACCESS = (BLOCK_SIZE*BLOCK_SIZE) * (MASK_WIDTH*MASK_WIDTH)
//
// - elementos carregados para cada TILE:
//
// INPUT_ACCESS = (BLOCK_SIZE + MASK_WIDTH - 1)^2
//
// - reduction:
//
// REDUCTION = TILE_ACCESS/INPUT_ACCESS
//
// ---------------------------------------------------------------
// MASK_WIDHT\BLOCK_SIZE |  8X8  | 14X14 | 15X15 | 16X16 | 32X32
// ---------------------------------------------------------------
//           5           | 11.11 | 15.12 | 15.58 | 16.00 | 19.75
//           7           | 16.00 | 24.01 | 25.00 | 25.92 | 34.75
//           9           | 20.25 | 32.80 | 34.45 | 36.00 | 51.84
//           11          | 23.90 | 41.17 | 43.56 | 45.82 | 70.24
//           13          | 27.04 | 49.00 | 52.16 | 55.18 | 89.39
// ---------------------------------------------------------------
//
// taf!
