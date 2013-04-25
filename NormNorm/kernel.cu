// Definitions for IntelliSense
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

// Thrust classes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
//#include <thrust/sort.h>
//#include <thrust/copy.h>
//#include <thrust/transform_reduce.h>
//#include <thrust/functional.h>

// Standard headers
#include <vector>
#include <stdio.h>
//#include <algorithm>
//#include <cstdlib>


// Constant memory is broadcasted to all threads (special caching)
#define N_GH 10

static __constant__ double c_rt2 = 1.4142135623730951;

static __constant__ double c_rt2pi = 2.5066282746310002;

static __constant__ double c_absc[] = {
	-3.4361591188377352, -2.5327316742327906, -1.756683649299881, -1.0366108297895147,
	-0.34290132722370503, 0.34290132722370431, 1.0366108297895129, 1.7566836492998816,
	2.5327316742327901, 3.4361591188377369 }; 

static __constant__ double c_wts[] = { 
	4.3106526307180023e-06, 0.00075807093431224217, 0.01911158050077038, 0.13548370298026716,
	0.34464233493201829, 0.34464233493201879, 0.13548370298026738, 0.019111580500770171,
	0.00075807093431222135, 4.3106526307190323e-06 }; 


// Functions used in kernels and CPU test codes
__device__ __host__
double rho(double x, double mu, double var)
{
	double t = x - mu;
	return exp(-t*t/(2.0*var));
}


// Kernel to derive the log marginals for each data item and parameters
__global__
void marginals(double *theta, int dim_theta, int n_theta, double *meas, double *meas_unc, int m, int n, double *marg)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int i_theta = blockDim.y * blockIdx.y + threadIdx.y;
	if (i<n && i_theta<n_theta)
	{
		double mu = theta[0+dim_theta*i_theta];
		double var = theta[1+dim_theta*i_theta];
		double x;
		double r;
		double mrg = 0;
		double mi = meas[i]; // here only 1 feature!!!
		double ui = meas_unc[i];
		for (int j=0; j<N_GH; j++) {
			x = mi + c_rt2 * ui * c_absc[j];
			r = rho(x, mu, var);
			mrg += c_wts[j] * r;
		}
		marg[i+n*i_theta] = log(mrg / c_rt2pi);
	}
}
	  

// Entry point
int main(int argc, char *argv[])
{
	// measurements
	int n = 20000000; // # of items
	int m = 1; // # of features

	// default GPU
	int devId = 0;

	// simple command line arguments
	if (argc > 1) n = atoi(argv[1]);
	if (argc > 2) devId = atoi(argv[2]);

	cudaError_t err = cudaSetDevice(devId);
	if (err != cudaSuccess) { return 1; }

	// Allocate data on CPU and load 
	thrust::host_vector<double> h_meas(n*m);
	thrust::host_vector<double> h_meas_unc(n*m);

	unsigned int seed = 98724732;
	double sigma_msmt = 3.;
	static thrust::minstd_rand rng(seed);
	thrust::random::experimental::normal_distribution<double> dist(0., sigma_msmt);

	double mu_popn_true = 20.;
	double sigma_popn_true = 1.;
	// Create simulated data; copies to GPU dumbly!
	for (int i=0; i<n; i++) {
		for (int j=0; j<m; j++) {
			h_meas[i*m+0] = mu_popn_true + dist(rng);
			h_meas_unc[i*m+1] = sigma_popn_true;
		}
	}

	// Copy data to GPU
	thrust::device_vector<double> d_meas = h_meas;
	thrust::device_vector<double> d_meas_unc = h_meas_unc;

	// Create array of hyperparameter values.
	// Here we condition on sigma_popn_true (for simple analytical result).
	// Currently ineffecient; should build host vector and copy over.
	int dim_theta = 2;
	int n_theta = 11;
	double sqrt_n = sqrt((double)n);
	double mu_lo = mu_popn_true - 2*sigma_msmt/sqrt_n;
	double mu_hi = mu_popn_true + 2*sigma_msmt/sqrt_n;
	double dmu = (mu_hi - mu_lo)/(n_theta-1.);
	double mu;
	thrust::host_vector<double> h_theta(dim_theta*n_theta);
	for (int i=0; i<n_theta; i++) {
		mu = mu_lo + i*dmu;
		h_theta[i*dim_theta] = mu;
		h_theta[i*dim_theta+1] = sigma_popn_true;
	}
	// copy to GPU
	thrust::device_vector<double> d_theta = h_theta;

	// To load a single set of thetas into constant memory, copy to a global:
	// cudaMemcpyToSymbol(c_theta, p_theta, d_theta.size() * sizeof(*p_theta));

	// Alloc mem for marginals for individuals, for each set of hyperparams.
	thrust::device_vector<double> d_marg(n*n_theta);

	// Here comes the GPU calculation
	{
		// log marginal likelhoods in parallel on all threads independently
		double* p_marg = thrust::raw_pointer_cast(&d_marg[0]);
		double* p_theta = thrust::raw_pointer_cast(&d_theta[0]);
		double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
		double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);

		// cuda grid launch
		dim3 nThreads(512,1);
		dim3 nBlocks((n + nThreads.x-1) / nThreads.x, (n_theta + nThreads.y-1) / nThreads.y);
		printf("nBlocks: %d  %d\n", nBlocks.x, nBlocks.y);  // no more than 64k blocks!
		if (nBlocks.x > 65535 || nBlocks.y > 65535) 
		{
			std::cerr << "ERROR: Block is too large" << std::endl;
			return 2;
		}
		marginals<<<nBlocks,nThreads>>>(p_theta, dim_theta, n_theta, p_meas, p_meas_unc, m, n, p_marg);

		// do anything here on the CPU?

		// wait for GPU to finish
		cudaError_t err = cudaDeviceSynchronize();

		/*
		// copy results back to host
		thrust::host_vector<double> h_marg = d_marg;
		for (int i=0; i<20; i++) {
		  printf("%d %20.10f \n", i, h_marg[i]);
		}
		*/

		// Reduction for all theta values - could use thrust::reduce_by_key() ?
		for (int i=0; i<n_theta; i++) {
			int start = i*n;
			int end = start + n;
			double log_marg = 0;
			log_marg = thrust::reduce(d_marg.begin()+start, d_marg.begin()+end);
			//std::cout << i << " " << log_marg << std::endl;
            printf("%d %20.10f %20.10f \n", i, log_marg, h_theta[i*dim_theta]);
		}

	}
	  
	return 0;
}
