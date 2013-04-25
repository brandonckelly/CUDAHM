#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random.h>

#include <stdio.h>

//#include <thrust/sort.h>
//#include <thrust/copy.h>

//#include <thrust/transform_reduce.h>
//#include <thrust/functional.h>

//#include <algorithm>
//#include <cstdlib>

#include <vector>
/*
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
 */

// Uniform random number transform, used for doing MHA acceptance step on the GPU
static __constant__ thrust::uniform_real_distribution<double> unif(0.0,1.0);

// Norm random number transform, used for generating new parameter proposals on the GPU
static __constant__ thrust::random::experimental::normal_distribution<double> snorm(0.0, 1.0);

// Target acceptance rate for Robust Adaptive Metropolis (RAM).
static __constant__ double target_rate = 0.4;

// Decay rate for proposal scaling factor updates. The proposal scaling factors decay as 1 / niter^decay_rate.
// This is gamma in the notation of Vihola (2012)
static __constant__ double decay_rate = 2.0 / 3.0;

// Current iteration of the MCMC sampler, needed to calculate the decay sequence for the RAM algorithm. Is it
// best to put this in constant memory on the GPU and update from the CPU every iteration?
static __constant__ int current_iter = 0


// Perform the MHA update on the n parameters, done in parallel on the GPU. This is what the kernel does.
__global__
void update_parameters(double* theta, double* parameters, double* data, double* sigmas, int n, double* logdens,
                       double* jump_sigma)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i<n)
	{
		double mu = theta[0];  // Get population parameters
		double var = theta[1];
        
        // Propose a new value for the parameters for this data point
        new_parameter = parameters[i] + jump_sigma[i] * snorm(rng);  // Not quite sure how to do this in parallel...
        
        // Compute the conditional log-posterior of the proposed parameter values for this data point
        double logdens_prop = -0.5 * (data[i] - new_parameter) * (data[i] - new_parameter) / (sigmas[i] * sigmas[i]) +
            0.5 * log(var) - 0.5 * (new_parameter - mu) * (new_parameter - mu) / var;
        
        // Compute the Metropolis ratio
        double ratio = logdens_prop - logdens[i];
        ratio = (ratio < 0.0) ? ratio : 0.0;  // Isn't there a min() function I can use to make this more obvious?
        ratio = exp(ratio);
        
        // Now randomly decide whether to accept or reject
        unif_draw = unif(rng);  // Not quite sure if this is the correct way to do this in parallel on the GPU...
        if (unif_draw < ratio) {
            // Accept this proposal, so save this parameter value and conditional log-posterior
            parameters[i] = new_parameter;
            logdens[i] = logdens_prop;
        }
        
        // Finally, adapt the scale of the proposal distribution
        // TODO: check that the ratio is finite before doing this step
        double decay_sequence = 1.0 / pow(niter, decay_rate)
        sigmas[i] *= exp(decay_sequence / 2.0 * (ratio - target_rate))
	}
}


int main(void)
{
	// measurements
	int n = 2000000; // # of items
	int m = 1; // # of features
	/*
     wrapvec d_features(m,n);
     wrapvec d_sigmas(m,n);
     */
	thrust::host_vector<double> h_features(n*m);
	thrust::host_vector<double> h_sigmas(n*m);
    
	unsigned int seed = 98724732;
	double sigma_msmt = 3.;
	static thrust::minstd_rand rng(seed);
	thrust::random::experimental::normal_distribution<double> dist(0., sigma_msmt);
    
	double mu_popn_true = 20.;
	double sigma_popn_true = 1.;
	// Create simulated data; copies to GPU dumbly!
	for (int i=0; i<n; i++) {
		for (int j=0; j<m; j++) {
			h_features[i*m+0] = mu_popn_true + dist(rng);
			h_sigmas[i*m+1] = sigma_popn_true;
		}
	}
	thrust::device_vector<double> d_features = h_features;
	thrust::device_vector<double> d_sigmas = h_sigmas;
    
	// Create array of hyperparameter values.
	// Here we condition on sigma_popn_true (for simple analytical result).
	// Currently ineffecient; should build host vector and copy over.
	int dim_theta = 2;
	int n_theta = 11;
	double mu_lo = mu_popn_true - 2*sigma_msmt/sqrt(n);
	double mu_hi = mu_popn_true + 2*sigma_msmt/sqrt(n);
	double dmu = (mu_hi - mu_lo)/(n_theta-1.);
	double mu;
	thrust::host_vector<double> h_theta(dim_theta*n_theta);
	for (int i=0; i<n_theta; i++) {
		mu = mu_lo + i*dmu;
		h_theta[i*dim_theta] = mu;
		h_theta[i*dim_theta+1] = sigma_popn_true;
	}
	thrust::device_vector<double> d_theta = h_theta;
    
	// To load a single set of thetas into constant memory, copy to a global:
	// cudaMemcpyToSymbol(c_theta, p_theta, d_theta.size() * sizeof(*p_theta));
    
	// Alloc mem for marginals for individuals, for each set of hyperparams.
	thrust::device_vector<double> d_marg(n*n_theta);
    
	{
		// log marginal likelhoods in parallel on all threads independently
		double* p_marg = thrust::raw_pointer_cast(&d_marg[0]);
		double* p_theta = thrust::raw_pointer_cast(&d_theta[0]);
		double* p_features = thrust::raw_pointer_cast(&d_features[0]);
		double* p_sigmas = thrust::raw_pointer_cast(&d_sigmas[0]);
        
		// cuda grid launch
		dim3 nThreads(32,8);
		dim3 nBlocks((n + nThreads.x-1) / nThreads.x, (n_theta + nThreads.y-1) / nThreads.y);
		printf("nBlocks: %d  %d\n", nBlocks.x, nBlocks.y);  // no more than 64k blocks!
		marginals<<<nBlocks,nThreads>>>(p_theta, dim_theta, n_theta,
                                        p_features, p_sigmas, m, n, p_marg);
		// wait for it to finish
		cudaError_t err = cudaDeviceSynchronize();
        
		thrust::host_vector<double> h_marg = d_marg;
		for (int i=0; i<20; i++) {
            printf("%d %20.10f \n", i, h_marg[i]);
		}
        
		// Loop over hyperparams; reduce over individuals for each case.
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
