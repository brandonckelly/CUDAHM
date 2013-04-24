#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>

//#include <thrust/sort.h>
//#include <thrust/copy.h>

//#include <thrust/transform_reduce.h>
//#include <thrust/functional.h>

//#include <algorithm>
//#include <cstdlib>

#include <vector>

typedef thrust::device_vector<double> dvec;
typedef std::vector<dvec> vdvec;

struct wrapvec
{
	vdvec v;

	wrapvec(int m, int n) : v(m)
	{
		for (int i=0; i<m; i++)	{			
			v[i].reserve(n);
		}
	}

	double** ptrs()
	{
		thrust::host_vector<double*> h_ptr(v.size());
		for (unsigned int i=0; i<v.size(); i++)	
			h_ptr[i] = (double*) thrust::raw_pointer_cast(&(v[i][0]));

		thrust::device_vector<double*> d_ptr = h_ptr;
		return (double**) thrust::raw_pointer_cast(&d_ptr[0]); 
	}
};

// constants for integration
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


__device__ __host__
double rho(double r)
{
	return 1;
}


__global__
void marginals(double *theta, int dim_theta, int n_theta, double **features, double **sigmas, int m, int n, double *marg)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int i_theta = blockDim.y * blockIdx.y + threadIdx.y;
	if (i<n && i_theta<n_theta)
	{
		double mu = theta[0+dim_theta*i_theta];
		double var = theta[1+dim_theta*i_theta];
		double x;
		double rho;
		marg[i] = 0;
		for (int j=0; j<N_GH; j++) {
			x = features[0][i] + c_rt2*sigmas[0][i]*c_absc[j];
			double t = x-mu;
			rho = exp(-t*t/(2.0*var))/c_rt2pi;
			marg[i+n*n_theta] += c_wts[j]*rho;
		}
	}
}
	  

int main(void)
{
	// measurements
	int n = 100; // # of items
	int m = 1; // # of features
	wrapvec d_features(m,n);
	wrapvec d_sigmas(m,n);

	unsigned int seed = 98724732;
	double sigma_msmt = 3.;
	static thrust::minstd_rand rng(seed);
	thrust::random::experimental::normal_distribution<double> dist(0., sigma_msmt);
	// thrust::generate(d_vec.begin(), d_vec.end(), dist(rng));

	double mu_popn_true = 20.;
	double sigma_popn_true = 1.;
	// Create simulated data; copies to GPU dumbly!
	for (int i=0; i<n; i++) {
		for (int j=0; j<m; j++) {
			d_features.v[j][i] = mu_popn_true + dist(rng);
			d_sigmas.v[j][i] = sigma_popn_true;
		}
	}

	// Create array of hyperparameter values.
	// Here we condition on sigma_popn_true (for simple analytical result).
	// Currently ineffecient; should build host vector and copy over.
	int dim_theta = 2;
	int n_theta = 10;
	double mu_lo = mu_popn_true - 5*sigma_msmt/sqrt(n);
	double mu_hi = mu_popn_true + 5*sigma_msmt/sqrt(n);
	double dmu = (mu_hi - mu_lo)/(n_theta-1.);
	double mu;
	thrust::device_vector<double> d_theta(dim_theta*n_theta);
	for (int i=0; i<n_theta; i++) {
		mu = mu_lo + i*dmu;
		d_theta[i*dim_theta] = mu;
		d_theta[i*dim_theta+1] = sigma_popn_true;
	}

	// To load a single set of thetas into constant memory, copy to a global:
	// cudaMemcpyToSymbol(c_theta, p_theta, d_theta.size() * sizeof(*p_theta));

	// Alloc mem for marginals for individuals, for each set of hyperparams.
	thrust::device_vector<double> d_marg(n*n_theta);

	{
		// log marginal likelhoods in parallel on all threads independently
		double* p_marg = thrust::raw_pointer_cast(&d_marg[0]);
		double* p_theta = thrust::raw_pointer_cast(&d_theta[0]);
		double** p_features = d_features.ptrs();
		double** p_sigmas = d_sigmas.ptrs();

		// cuda grid launch
		dim3 nThreads(16,16);
		dim3 nBlocks((n + nThreads.x-1) / nThreads.x, (n_theta + nThreads.y-1) / nThreads.y);
		marginals<<<nBlocks,nThreads>>>(p_theta, dim_theta, n_theta, 
			p_features, p_sigmas, m, n, p_marg);
		// wait for it to finish
		cudaError_t err = cudaDeviceSynchronize();

		// Loop over hyperparams; reduce over individuals for each case.
		for (int i=0; i<n_theta; i++) {
			int start = i*n;
			int end = start + n;
			double log_marg = thrust::reduce(d_marg.begin()+start, d_marg.begin()+end);
			std::cout << log_marg << std::endl;
		}

	}
	  
	return 0;

}
