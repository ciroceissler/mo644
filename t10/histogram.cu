#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

#define HISTOGRAM_SIZE 64

typedef struct {
	unsigned char red, green, blue;
} PPMPixel;

typedef struct {
	int x, y;
	PPMPixel *data;
} PPMImage;

__global__ void histo_kernel(PPMPixel *pixel, int size, float *histo) {
    __shared__ float histo_private[HISTOGRAM_SIZE];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    if (threadIdx.x < HISTOGRAM_SIZE) {
        histo_private[threadIdx.x] = 0;
    }

    __syncthreads();

    while (i < size) {
        atomicAdd(&(histo_private[pixel[i].red*16 + pixel[i].green*4 + pixel[i].blue]), 1);

        i += stride;
    }

    __syncthreads();

    if (threadIdx.x < HISTOGRAM_SIZE) {
        atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
    }

}

__global__ void histo_normalization(float *histo, int n) {
    histo[threadIdx.x] = histo[threadIdx.x]/n;
}

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


void Histogram(PPMImage *image, float *h) {

    int i;
	int rows, cols;

	float n = image->y * image->x;

    // begin - cuda variables
    float *d_h;
    PPMPixel *d_pixel;

    cudaDeviceProp prop;

    cudaEvent_t p_1, p_2, p_3, p_4, p_5;
    float milliseconds_1 = 0;
    float milliseconds_2 = 0;
    float milliseconds_3 = 0;
    float milliseconds_4 = 0;
    // end - cuda variables

	cols = image->x;
	rows = image->y;

	for (i = 0; i < n; i++) {
		image->data[i].red = floor((image->data[i].red * 4) / 256);
		image->data[i].blue = floor((image->data[i].blue * 4) / 256);
		image->data[i].green = floor((image->data[i].green * 4) / 256);
	}

    // begin - cuda execution
    cudaEventCreate(&p_1);
    cudaEventCreate(&p_2);
    cudaEventCreate(&p_3);
    cudaEventCreate(&p_4);
    cudaEventCreate(&p_5);
    cudaGetDeviceProperties(&prop, 0);

    cudaEventRecord(p_1);

    cudaMalloc((void **)&d_pixel, sizeof(unsigned char)*3*rows*cols);
    cudaMalloc((void **)&d_h, sizeof(float)*HISTOGRAM_SIZE);

    cudaEventRecord(p_2);

    cudaMemcpy(d_pixel, image->data, sizeof(unsigned char)*3*rows*cols, cudaMemcpyHostToDevice);
    cudaMemset(d_h, 0,HISTOGRAM_SIZE*sizeof(float));

    cudaEventRecord(p_3);

    histo_kernel<<<2*prop.multiProcessorCount, HISTOGRAM_SIZE>>>(d_pixel, cols*rows, d_h);
    histo_normalization<<<1, 64>>>(d_h, cols*rows);

    cudaEventRecord(p_4);

    cudaMemcpy(h, d_h, sizeof(float)*HISTOGRAM_SIZE, cudaMemcpyDeviceToHost);

    cudaEventRecord(p_5);

    cudaFree(d_h);
    cudaFree(d_pixel);
    // end - cuda execution
    cudaEventSynchronize(p_4);

    cudaEventElapsedTime(&milliseconds_1, p_1, p_2);
    cudaEventElapsedTime(&milliseconds_2, p_2, p_3);
    cudaEventElapsedTime(&milliseconds_3, p_3, p_4);
    cudaEventElapsedTime(&milliseconds_4, p_4, p_5);
}

int main(int argc, char *argv[]) {

	if( argc != 2 ) {
		printf("Too many or no one arguments supplied.\n");
	}

	int i;
	char *filename = argv[1]; //Recebendo o arquivo!;
	
	//scanf("%s", filename);
	PPMImage *image = readPPM(filename);

	float *h = (float*)malloc(sizeof(float) * HISTOGRAM_SIZE);

	//Inicializar h
	for(i=0; i < HISTOGRAM_SIZE; i++) h[i] = 0.0;

	Histogram(image, h);

	for (i = 0; i < HISTOGRAM_SIZE; i++){
		printf("%0.3f ", h[i]);
	}
	printf("\n");
	//fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);  
	free(h);
}

// NOTES(ciroceissler):
//
// arq1.ppm, 0.279246s, 0.318880ms, 8.368736 ms, 1.688352 ms, 0.017504ms, 10.393472ms, 26.8674414
// arq2.ppm, 0.559345s, 0.291424ms, 18.400768ms, 6.097952 ms, 0.015392ms, 24.805536ms, 22.5492003
// arq3.ppm, 2.127529s, 0.298560ms, 72.371140ms, 22.600864ms, 0.015072ms, 95.285636ms, 22.3279089
