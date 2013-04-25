#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random.h>

#include <stdio.h>
#include <stdlib.h>

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
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)

// Target acceptance rate for Robust Adaptive Metropolis (RAM).
static __constant__ double c_target_rate = 0.4;

// Decay rate for proposal scaling factor updates. The proposal scaling factors decay as 1 / niter^decay_rate.
// This is gamma in the notation of Vihola (2012)
static __constant__ double c_decay_rate = 0.66667;

// Current iteration of the MCMC sampler, needed to calculate the decay sequence for the RAM algorithm. Is it
// best to put this in constant memory on the GPU and update from the CPU every iteration?
static __constant__ int c_current_iter = 0;

// Pointer to the current value of theta. This is stored in constant memory so that all the threads on the GPU
// can access the same theta quickly.
static __constant__ double c_theta[2];

// Initialize the random number generator state
__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
     number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

// Perform the MHA update on the n parameters, done in parallel on the GPU. This is what the kernel does.
__global__
void update_chi(double* theta, double* chi, double* meas, double* meas_unc, int n, double* logdens, double* jump_sigma,
                curandState* state)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i<n)
	{
		double mu = theta[0];  // Get population parameters
		double var = theta[1];
        
        double meas_i = meas[i];  // Get measurements
        double meas_unc_i = meas_unc[i];
        
        /* Copy state to local memory for efficiency */
        curandState localState = state[i];
        
        // Propose a new value of the characteristics for this data point
        double new_chi = chi[i] + jump_sigma[i] * curand_normal_double(&localState);

        // Compute the conditional log-posterior of the proposed parameter values for this data point
        double logdens_prop = -0.5 * (meas_i - new_chi) * (meas_i - new_chi) / (meas_unc_i * meas_unc_i) +
            0.5 * log(var) - 0.5 * (new_chi - mu) * (new_chi - mu) / var;
        
        // Compute the Metropolis ratio
        double ratio = logdens_prop - logdens[i];
        ratio = (ratio < 0.0) ? ratio : 0.0;
        ratio = exp(ratio);
        
        // Now randomly decide whether to accept or reject
        double unif_draw = curand_uniform_double(&localState);
        
        if (unif_draw < ratio) {
            // Accept this proposal, so save this parameter value and conditional log-posterior
            chi[i] = new_chi;
            logdens[i] = logdens_prop;
        }

        // Copy state back to global memory
        state[i] = localState;

        // Finally, adapt the scale of the proposal distribution
        // TODO: check that the ratio is finite before doing this step
        double decay_sequence = 1.0 / pow(c_current_iter, c_decay_rate);
        jump_sigma[i] *= exp(decay_sequence / 2.0 * (ratio - c_target_rate));
	}
}


int main(void)
{
	int n = 2000; // # of data points
	int m = 1; // # of features per data point (e.g., # of points in the SED for the i^th source)
    int p = 1; // * of characteristics per data point

    // Random number generator and distribution needed to simulate some data
	unsigned int seed = 98724732;
	static thrust::minstd_rand rng(seed);
	thrust::random::experimental::normal_distribution<double> snorm(0., 1.0);
    
    // Population level parameters: theta, where theta parameterizes the distribution of chi
    int dim_theta = 2;
	double mu_popn_true = 20.;  // Average value of the chi values
	double sigma_popn_true = 1.;  // Standard deviation of the chi values
    thrust::host_vector<double> h_theta(dim_theta);  // Allocate memory on host
    h_theta[0] = mu_popn_true;
    h_theta[1] = sigma_popn_true * sigma_popn_true;  // theta = (mu,var)
    
    // Allocate memory for arrays on host
    thrust::host_vector<double> h_meas(n * m);  // The measurements, m values for each of n data points
	thrust::host_vector<double> h_meas_unc(n * m);  // The measurement uncertainties
    thrust::host_vector<double> h_chi(n * p);  // Unknown characteristics, p values for each of n data points
    
    // Scale of proposal distribution for each data point, used by the Metropolis algorithm. When p > 1 this will be
    // the Cholesky factor of the proposal covariance matrix.
    thrust::host_vector<double> h_jump_sigma(n);
    
	// Create simulated characteristics and data
    std::vector<double> true_chi(n * p);
    double sigma_msmt = 3.;  // Standard deviation for the measurement errors
    
	for (int i=0; i<n; i++) {
		for (int j=0; j<p; j++) {
            // First generate true value of the characteristics
			true_chi[i * p + j] = mu_popn_true + sigma_popn_true * snorm(rng);
            // Initialize the scale of the chi proposal distributions to just be the measurement uncertainty
            h_jump_sigma[i * m + j] = sigma_msmt;
        }
        for (int k=0; k<m; k++) {
            // Now generate measurements given the true characteristics
			h_meas_unc[i * m + k] = sigma_msmt;
            // Just assume E(meas|chi) = chi for now
            h_meas[i * m + k] = true_chi[i * p + k] + sigma_msmt * snorm(rng);
		}
	}
    
    // Allocate memory for arrays on device and copy the values from the host
    thrust::device_vector<double> d_meas = h_meas;
	thrust::device_vector<double> d_meas_unc = h_meas_unc;
    thrust::device_vector<double> d_chi = h_chi;
    thrust::device_vector<double> d_jump_sigma = h_jump_sigma;
    thrust::device_vector<double> d_logdens;  // log posteriors for an individual data point
    
	// Load a single set of thetas into constant memory
    double* p_theta = thrust::raw_pointer_cast(&h_theta[0]);
	CUDA_CALL(cudaMemcpyToSymbol(c_theta, p_theta, h_theta.size() * sizeof(*p_theta)));
    
    // Cuda grid launch
    dim3 nThreads(256);
    dim3 nBlocks((n + nThreads.x-1) / nThreads.x);
    printf("nBlocks: %d\n", nBlocks.x);  // no more than 64k blocks!
    if (nBlocks.x > 65535)
    {
        std::cerr << "ERROR: Block is too large" << std::endl;
        return 2;
    }

    curandState* devStates;  // Create state object for random number generators on the GPU
    // Allocate memory on GPU for RNG states
    CUDA_CALL(cudaMalloc((void **)&devStates, nThreads.x * nBlocks.x *
                         sizeof(curandState)));
    // Initialize the random number generator states on the GPU
    setup_kernel<<<nBlocks,nThreads>>>(devStates);

    // Wait until everything is done running on the GPU, make sure everything went OK
    CUDA_CALL(cudaDeviceSynchronize());

    // Now run the MCMC sampler
    
    /*
	{
		// log marginal likelhoods in parallel on all threads independently
		double* p_marg = thrust::raw_pointer_cast(&d_marg[0]);
		double* p_theta = thrust::raw_pointer_cast(&d_theta[0]);
		double* p_features = thrust::raw_pointer_cast(&d_features[0]);
		double* p_sigmas = thrust::raw_pointer_cast(&d_sigmas[0]);
        
		//marginals<<<nBlocks,nThreads>>>(p_theta, dim_theta, n_theta, p_features, p_sigmas, m, n, p_marg);
		// wait for it to finish
		//cudaDeviceSynchronize();
        
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
    */
    cudaFree(devStates);  // Free up the memory on the GPU from the RNG states
    
	return 0;
}
