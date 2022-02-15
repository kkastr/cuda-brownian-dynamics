#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

__global__ void rng_setup_kernel(unsigned int seed,curandStatePhilox4_32_10_t *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(seed, id, 0, &state[id]);
}


__device__ void integration_kernel(float dt, int steps, float *x, float *y, float *z,curandStatePhilox4_32_10_t *state)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  float xrn;
  float yrn;
  float zrn;
  float updf;
  float prob;

  curandStatePhilox4_32_10_t localState = state[i];

  xrn = curand_uniform(&localState)-0.5;
  yrn = curand_uniform(&localState)-0.5;
  zrn = curand_uniform(&localState)-0.5;

  state[i] = localState;

  x[i] = x[i] + prf*xrn*dt;
  y[i] = y[i] + prf*yrn*dt;
  z[i] = z[i] + prf*zrn*dt;

  float r = sqrt(x[i]*x[i] + y[i]*y[i]);

  //TODO: fix boundary conditions
  if (r>rcav){
    x[i] = x[i]*rcav/r;
    y[i] = y[i]*rcav/r;
  }
  if( z[i] < zmin){
    z[i] = zmin;
  }
  if( z[i] > lcav){
    z[i] = lcav-(float)1;
  }
}


int main(int argc, char* argv[])
{
  unsigned int seed;
  double tmax;

  int kernel_typeflag;

  if (argc > 1){
		seed = atoi(argv[0]); //sim seed
    tmax = atof(argv[1]); //maximum runtime in timesteps;
	}
	else {
		printf("No arguments given. \n Need to provide the following: seed, tmax.\n");
		return 1;
	}

  srand(seed);
  double t = 0;
  int steps = 0;
  float dt = 0.01;
  int outputfreq = 1000;
  float kT = 1, m = 1, gamma = 1;


  float prf = sqrt((2*kT*gamma)/(m*dt));
  float *hx, *hy, *hz, *d_x, *d_y, *d_z;

  char cout_pos[64];
	sprintf(cout_pos,"trajectory.xyz";

  FILE *cout_position;
  cout_position=fopen(cout_pos,"w");

  const unsigned int threadsPerBlock = 64;
  const unsigned int blockCount = 64;
  int N = threadsPerBlock * blockCount;

  hx = (float*)malloc(N*sizeof(float));
  hy = (float*)malloc(N*sizeof(float));
  hz = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));
  cudaMalloc(&d_z, N*sizeof(float));

  curandStatePhilox4_32_10_t *devPHILOXStates;

  cudaMalloc((void **)&devPHILOXStates, N*sizeof(curandStatePhilox4_32_10_t));

  rng_setup_kernel<<<blockCount, threadsPerBlock>>>(seed,devPHILOXStates);

  for (int j=0;j<N;j++){
    hx[j] = 0.0f;
    hy[j] = 0.0f;
    hz[j] = 0.0f;

  }

  cudaMemcpy(d_x, hx, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, hy, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, hz, N*sizeof(float), cudaMemcpyHostToDevice);

  while (t < tmax)
  {
    //TODO: fix inputs
    integration_kernel<<<blockCount, threadsPerBlock>>>(kernel_typeflag, dt, steps, d_x, d_y, d_z, devPHILOXStates);


    if (steps%outputfreq==0)
    {
        cudaMemcpy(hx, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hy, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hz, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);

      fprintf(cout_position, "%i", N)
      fprintf(cout_position, "comment")
      for (int i = 0; i < N; i++)
      {
        fprintf(cout_position,"%i,%f,%f,%f\n",i,hx[i],hy[i],hz[i]);

      }
    }

    t+=dt;
    steps++;
  }
  cudaFree(devPHILOXStates);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  free(hx);
  free(hy);
  free(hz);
}
