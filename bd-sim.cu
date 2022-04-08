#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

__global__ void rng_setup_kernel(unsigned int seed, curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
            number, no offset */
    curand_init(seed, id, 0, &state[id]);
}

__global__ void integration_kernel(int N, float dt, float prf, float intTime, float lxHalf, float lyHalf, float lzHalf, float *x, float *y, float *z, curandState *state) {

    float xrn;
    float yrn;
    float zrn;

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < N) {
        for (int t=0; t < intTime; t++) {
            xrn = curand_uniform(&state[i])-0.5;
            yrn = curand_uniform(&state[i])-0.5;
            zrn = curand_uniform(&state[i])-0.5;

            x[i] = x[i] + prf * xrn * dt;
            y[i] = y[i] + prf * yrn * dt;
            z[i] = z[i] + prf * zrn * dt;

            if (abs(x[i]) > lxHalf) {
                int sgnx = x[i]/abs(x[i]);
                x[i] = x[i] - 2 * sgnx * (abs(x[i]) - lxHalf);
            }
            if (abs(y[i]) > lyHalf) {
                int sgny = y[i]/abs(y[i]);
                y[i] = y[i] - 2 * sgny * (abs(y[i]) - lyHalf);
            }
            if (abs(z[i]) > lzHalf) {
                int sgnz = z[i]/abs(z[i]);
                z[i] = z[i] - 2 * sgnz * (abs(z[i]) - lzHalf);
            }
        }
    }
}

float uniform_distribution(int rangeLow, int rangeHigh) {
    float randVal = (float)rand() / (float)RAND_MAX;
    int range = rangeHigh - rangeLow + 1;
    float randVal_scaled = (randVal * (float)range) + (float)rangeLow;
    return randVal_scaled;
}


int arraySum_int (int arr[], int arrLength) {

    int sum = 0;

    for (int i = 0; i < arrLength; i++) {
        sum += arr[i];
    }

    return sum;
}

static void printTrajectory(int Npart, FILE *fp, int frame, float x[], float y[], float z[]) {


    for (int i=0; i < Npart; i++) {
        fprintf(fp, "%i,%i,%f,%f,%f\n", frame, i, x[i], y[i], z[i]);
    }
}

int main(int argc, char* argv[]) {
    unsigned int seed;
    int gpu_id;
    int N;
    int lbox;
    int maxtime;

    if (argc > 1){
        gpu_id = atoi(argv[1]); // which gpu to run on
        seed = atoi(argv[2]); //sim seed
        N = atoi(argv[3]); // number of particles
        lbox = atoi(argv[4]); // side length of the box
        maxtime = (int)atof(argv[5]); //maximum simulation steps to run the integration

    } else {
        printf("No arguments given.\nNeed to provide the following: gpu_id, seed, number of particles, box side length, max amount of time for the sim to run.\n");
        return 1;
    }

    cudaSetDevice(gpu_id);
    srand(seed);

    int steps = 0;
    int outfreq = 100;
    int maxsteps = maxtime / outfreq;
    float dt = 0.1f;

    float lxHalf = lbox / 2;
    float lyHalf = lbox / 2;
    float lzHalf = lbox / 2;

    float kT = 1.0f, m = 1.0f, gamma = 1.0f;
    float D, prf;

    D = kT*gamma/m;

    prf = sqrt(12.0f * 2.0f * D / dt);

    curandState *devStates;
    float *hx, *hy, *hz, *dx, *dy, *dz;

    char cout_filename[64];
    FILE *cout_fp;

    sprintf(cout_filename, "trajectory-N%i-frames%i-lbox%i-seed%i.csv", N, maxsteps, lbox, seed);
    cout_fp = fopen(cout_filename, "w");

    fprintf(cout_fp, "frame,id,x,y,z\n");

    hx = (float*)malloc(N*sizeof(float));
    hy = (float*)malloc(N*sizeof(float));
    hz = (float*)malloc(N*sizeof(float));

    for (int i = 0; i < N; i++) {
        // hx[i] = uniform_distribution(-lxHalf, lxHalf);
        // hy[i] = uniform_distribution(-lyHalf, lyHalf);
        // hz[i] = uniform_distribution(-lzHalf, lzHalf);
        hx[i] = 0.0f;
        hy[i] = 0.0f;
        hz[i] = 0.0f;

    }

    cudaMalloc(&dx, N * sizeof(float));
    cudaMalloc(&dy, N * sizeof(float));
    cudaMalloc(&dz, N * sizeof(float));

    cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dz, hz, N * sizeof(float), cudaMemcpyHostToDevice);

    const unsigned int threadsPerBlock = 256;
    const unsigned int blockCount = ceil((float)N / (float)threadsPerBlock);

    cudaMalloc((void **)&devStates, N * sizeof(curandState));

    rng_setup_kernel<<<blockCount, threadsPerBlock>>>(seed, devStates);

    while (steps < maxsteps) {

        cudaMemcpy(hx, dx, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hy, dy, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hz, dz, N * sizeof(float), cudaMemcpyDeviceToHost);

        printTrajectory(N, cout_fp, steps, hx, hy, hz);

        integration_kernel<<<blockCount, threadsPerBlock>>>(N, dt, prf, outfreq, lxHalf, lyHalf, lzHalf, dx, dy, dz, devStates);

        if (cudaDeviceSynchronize() != cudaSuccess) {
            fprintf (stderr, "Cuda call failed\n");
        }

        steps++;
    }

    cudaFree(devStates);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    free(hx);
    free(hy);
    free(hz);
}
