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


// fix inputs to kernel
__device__ void integration_kernel(float dt, int steps, float rcav, float lcav, float zmin, float rtarget, float p0, float prf, int *cap, int *atmp, int *captime, float *x, float *y, float *z,curandStatePhilox4_32_10_t *state)
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


// TODO: rename vars on host so that it is more clear they are not at all shared with the device kernel vars.
int main(int argc, char* argv[])
{
  const char *simtype;
  unsigned int seed;
  double tmax;
  float rtarget;
	float p0;
  int lp;
  int lc;

  int kernel_typeflag;


  if (argc > 1){
		seed = atoi(argv[2]); //sim seed
    tmax = atof(argv[3]); //maximum runtime in timesteps;
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
  float *x, *y, *z, *d_x, *d_y, *d_z;

  char cout_pos[64];
	sprintf(cout_pos,"trajectory.xyz";

  FILE *cout_position;
  cout_position=fopen(cout_pos,"w");

  const unsigned int threadsPerBlock = 64;
  const unsigned int blockCount = 64;
  int N = threadsPerBlock * blockCount;

  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  z = (float*)malloc(N*sizeof(float));

  cap = (int*)malloc(N*sizeof(int));
  atmp = (int*)malloc(N*sizeof(int));
  captime = (int*)malloc(N*sizeof(int));

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));
  cudaMalloc(&d_z, N*sizeof(float));

  curandStatePhilox4_32_10_t *devPHILOXStates;

  cudaMalloc((void **)&devPHILOXStates, N*sizeof(curandStatePhilox4_32_10_t));

  rng_setup_kernel<<<blockCount, threadsPerBlock>>>(seed,devPHILOXStates);

  for (int j=0;j<N;j++){
    x[j] = 10.0f;
    y[j] = 10.0f;
    z[j] = 10.0f;

  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, N*sizeof(float), cudaMemcpyHostToDevice);

  while (t < tmax)
  {
    //TODO: fix inputs
    integration_kernel<<<blockCount, threadsPerBlock>>>(kernel_typeflag, dt, steps, rcav, lcav, zmin, rtarget, p0, prf, d_cap, d_atmp, d_captime, d_x, d_y, d_z, devPHILOXStates);


    if (steps%outputfreq==0)
    {
        cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);

      fprintf(cout_position, "%i", N)
      fprintf(cout_position, "comment")
      for (int i = 0; i < N; i++)
      {
        fprintf(cout_position,"%i,%f,%f,%f\n",i,x[i],y[i],z[i]);

      }
    }

    t+=dt;
    steps++;
  }
  cudaFree(devPHILOXStates);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  free(x);
  free(y);
  free(z);
}
