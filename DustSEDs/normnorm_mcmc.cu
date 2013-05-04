// Cuda Includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

// Standard includes
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <vector>

//#include <thrust/sort.h>
//#include <thrust/copy.h>

//#include <algorithm>
//#include <cstdlib>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)

// Target acceptance rate for Robust Adaptive Metropolis (RAM).
static __constant__ double c_target_rate = 0.4;

// Decay rate for proposal scaling factor updates. The proposal scaling factors decay as 1 / niter^decay_rate.
// This is gamma in the notation of Vihola (2012)
static __constant__ double c_decay_rate = 0.66667;

// Pointer to the current value of theta. This is stored in constant memory so that all the threads on the GPU
// can access the same theta quickly.
static __constant__ double c_theta[6];

// Constants used for the model SED
static __constant__ double c_nu0 = 2.3e11;  // nu0 = 230 GHz
static const double nu0 = 2.3e11;
static const double hplanck = 6.6260755e-27;  // Planck's constant, in cgs
static const double clight = 2.997925e10;
static const double kboltz = 1.380658e-16;
static __constant__ double c_2h_over_csqr = 1.4745002e-47;  // 2 * hplanck / clight^2, used in the Planck function
static __constant__ double c_h_over_k = 4.7992157e-11;  // hplanck / kboltz

// Function to return the model SED at a input frequency, given the characteristics chi = (norm, beta, temp)
__device__
double modified_bbody(double nu, double normalization, double beta, double temperature)
{
    double bbody_numer = c_2h_over_csqr * nu * nu * nu;
    double bbody_denom = exp(c_h_over_k * nu / temperature) - 1.0;
    double bbody = bbody_numer / bbody_denom;
    double nu_over_nu0 = nu / c_nu0;
    double SED = normalization * pow(nu_over_nu0, beta) * bbody;
    return SED;
}


// Initialize the random number generator state
__global__
void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
     number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

// Perform the MHA update on the n parameters, done in parallel on the GPU. This is what the kernel does.
__global__
void update_chi(double* chi, double* meas, double* meas_unc, int n, double* logdens, double* jump_sigma,
                curandState* state, int current_iter)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i<n)
	{
        double mu = c_theta[0];  // Get population parameters
        double var = c_theta[1];
        
        double meas_i = meas[i];  // Get measurements
        double meas_unc_i = meas_unc[i];
        
        /* Copy state to local memory for efficiency */
        curandState localState = state[i];
        
        // Propose a new value of the characteristics for this data point
        double new_chi = chi[i] + jump_sigma[i] * curand_normal_double(&localState);

        // Compute the conditional log-posterior of the proposed parameter values for this data point
        double centered_meas = meas_i - new_chi;
        double logdens_meas = -0.5 * centered_meas * centered_meas / (meas_unc_i * meas_unc_i);
        double centered_chi = new_chi - mu;
        double logdens_pop = -0.5 * log(var) - 0.5 * centered_chi * centered_chi / var;
        double logdens_prop = logdens_meas + logdens_pop;
        
        // Compute the Metropolis ratio
        double ratio = logdens_prop - logdens[i];
        ratio = (ratio < 0.0) ? ratio : 0.0;
        ratio = exp(ratio);
        
        // Now randomly decide whether to accept or reject
        double unif_draw = curand_uniform_double(&localState);
        
        if (unif_draw < ratio) {
            // Accept this proposal, so save this parameter value and conditional log-posterior
            chi[i] = new_chi;
            logdens[i] = logdens_pop;
        }

        // Copy state back to global memory
        state[i] = localState;

        // Finally, adapt the scale of the proposal distribution
        // TODO: check that the ratio is finite before doing this step
        double decay_sequence = 1.0 / pow(current_iter, c_decay_rate);
        jump_sigma[i] *= exp(decay_sequence / 2.0 * (ratio - c_target_rate));
	}
}

struct zsqr {
    double mu, var;
    zsqr(double m, double v) : mu(m), var(v) {}
    
    __device__ __host__
    double operator()(double chi) {
        double chi_cent = chi - mu;
        double chisqr = -0.5 * chi_cent * chi_cent / var;
        return chisqr;
    }
};


int main(void)
{
    // Random number generator and distribution needed to simulate some data
	// unsigned int seed = 98724732;
    unsigned int seed =123455;
	static thrust::minstd_rand rng(seed);
	thrust::random::experimental::normal_distribution<double> snorm(0., 1.0);
    thrust::random::uniform_real_distribution<double> uniform(0.0, 1.0);
    
    /*
     Population level parameters: theta, where theta parameterizes the distribution of chi.
     For this application, chi = (log10(norm), beta, log10(temp)), where the model SED is a
     modified black body:
    
        SED(nu) = norm * (nu / nu0)^beta * Bbody(nu,temp).
     
     Here, Bbody(nu,temp) is the Planck function. For simplicity, right now I just assume that
     log10(norm), beta, and log10(temp) are independently drawn from different Gaussian distributions,
     although we should later include correlations among the parameters. So, for now, theta is just the
     collection of the mean and variances of the normal distributions.
    */
    
    int n = 2000; // # of data points
	int m = 5; // # of features per data point (i.e., # of points in the SED for the i^th source)
    int p = 3; // # of characteristics per data point.
    int dim_theta = 2 * p;
    
    double mu_norm = 8.5 * log(10);  // Average value of the natural logarithm of the SED normalization
    double sig_norm = 0.5 * log(10);  // Standard deviation in the SED normalization
    double mu_beta = 2.0;  // Average value of the SED power-law index
    double sig_beta = 0.2;  // Standard deviation in the SED power-law index
    double mu_temp = log(15.0);  // Average value of the logarithm of the dust temperature
    double sig_temp = 0.2;  // Standard deviation in the logarithm of the SED temperature
    
    thrust::host_vector<double> h_theta(dim_theta);  // Allocate memory on host
    h_theta[0] = mu_norm;
    h_theta[1] = mu_beta;
    h_theta[2] = mu_temp;
    h_theta[3] = sig_norm * sig_norm;
    h_theta[4] = sig_beta * sig_beta;
    h_theta[5] = sig_temp * sig_temp;
    
    // Allocate memory for arrays on host
    thrust::host_vector<double> h_meas(n * m);  // The measurements, m values for each of n data points
	thrust::host_vector<double> h_meas_unc(n * m);  // The measurement uncertainties
    thrust::host_vector<double> h_chi(n * p);  // Unknown characteristics, p values for each of n data points
    
    // Cholesky factor of proposal distribution for each data point, used by the Metropolis algorithm. The proposal
    // covariance matrix for each data point is Covar_i = PropChol_i * transpose(PropChol_i).
    dim_cholfactor = p * p - ((p - 1) * p) / 2  // only need one of the off-diagonal terms
    thrust::host_vector<double> h_prop_cholfact(n * dim_chol_factor);
    
	// Create simulated characteristics and data
    thrust::host_vector<double> h_true_chi(n * p);
    double sigma_msmt = [2.2e-4, 3.3e-4, 5.2e-4, 3.7e-4, 2.2e-4];  // Standard deviation for the measurement errors
    double lambda = [70.0, 170.0, 250.0, 350.0, 500.0];  // frequency bands correspond to Herschel PACS and SPIRES wavelengths
    double frequencies[5];
    for (int k=0; k<m; k++) {
        frequencies[k] = 1e6 / lambda[i];
    }
    
	for (int i=0; i<n; i++) {
        // Loop over the data indices
		for (int j=0; j<p; j++) {
            // Loop over the chi indices to generate true value of the characteristics
			h_true_chi[i + n * j] = h_theta[j] + sqrt(h_theta[p + j]) * snorm(rng);
            // Initialize the scale of the chi proposal distributions to just be some small constant value
            h_jump_sigma[i + n * j] = 0.01;
        }
        for (int k=0; k<m; k++) {
            // Loop over the feature indices to generate measurements, given the characteristics
            double bbody_numer = 2.0 * hplanck * frequencies[k] * frequencies[k] * frequencies[k] / (clight * clight);
            double temperature = exp(h_true_chi[n * 2 + i]); // Grap the temperature for this data point
            double bbody_denom = exp(hplanck * frequencies[k] / (kboltz * temperature)) - 1.0;
            double bbody = bbody_numer / bbody_denom;
            // Grab the SED normalization and power-law index
            double normalization = exp(h_true_chi[i]);
            double beta = h_true_chi[n + i];
            
            double nu_over_nu0 = frequencies[k] / nu0
            double SED_ik = normalization * pow(nu_over_nu0, beta) * bbody;
            
			h_meas_unc[k * n + i] = sigma_msmt[k];
            // Just assume E(meas|chi) = chi for now
            h_meas[k * n + i] = SED_ik + sigma_msmt[k] * snorm(rng);
		}
	}
    
    // Initialize the chis to their true values for now
    thrust::copy(h_true_chi.begin(), h_true_chi.end(), h_chi.begin());
    
    // Allocate memory for arrays on device and copy the values from the host
    thrust::device_vector<double> d_meas = h_meas;
    thrust::device_vector<double> d_meas_unc = h_meas_unc;
    thrust::device_vector<double> d_chi = h_chi;
    thrust::device_vector<double> d_prop_cholfact = h_prop_cholfact;
    thrust::device_vector<double> d_logdens(n);  // log posteriors for an individual data point
    
    thrust::fill(d_logdens.begin(), d_logdens.end(), -1.0e300);
    
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

    std::ofstream chifile("chis.dat");
    std::ofstream thetafile("thetas.dat");
    
    int mcmc_iter = 1000;
    int naccept_theta = 0;
    for (int i=0; i<mcmc_iter; i++) {
        // Now grab the pointers to the vectors, needed to run the kernel since it doesn't understand Thrust
        // We do this here because the thrust vector are smart, and we want to make sure they don't reassign
        // memory for whatever reason. This is very cheap to do, so better safe than sorry.
        double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
        double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
        double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
        double* p_jump_sigma = thrust::raw_pointer_cast(&d_jump_sigma[0]);
        double* p_logdens = thrust::raw_pointer_cast(&d_logdens[0]);
        int current_iter = i + 1;
        update_chi<<<nBlocks,nThreads>>>(p_chi, p_meas, p_meas_unc, n, p_logdens, p_jump_sigma, devStates, current_iter);
        
        // Generate new theta in parallel with GPU calculation above
        double proposed_theta[2];
        proposed_theta[0] = h_theta[0] + 0.1 * snorm(rng);
        proposed_theta[1] = h_theta[1] + 0.1 * snorm(rng);
        
        CUDA_CALL(cudaDeviceSynchronize());
        
        double logdens_pop = thrust::transform_reduce(d_chi.begin(), d_chi.end(), zsqr(proposed_theta[0], proposed_theta[1]), 0.0, thrust::plus<double>());
        logdens_pop += -n / 2.0 * log(proposed_theta[1]);
        
        double logdens_old = thrust::reduce(d_logdens.begin(), d_logdens.end());
        
        double lograt = logdens_pop - logdens_old;
        lograt = std::min(lograt, 0.0);
        double ratio = exp(lograt);
        double unif = uniform(rng);
        
        if (unif < ratio) {
            h_theta[0] = proposed_theta[0];
            h_theta[1] = proposed_theta[1];
            CUDA_CALL(cudaMemcpyToSymbol(c_theta, p_theta, h_theta.size() * sizeof(*p_theta)));
            naccept_theta++;
        }
        
        thetafile << h_theta[0] << " " << h_theta[1] << std::endl;

        std::cout << current_iter << std::endl;
        thrust::copy(d_chi.begin(), d_chi.end(), h_chi.begin());
        for (int j=0; j<n; j++){
            chifile << " " << h_chi[j];
        }
        chifile << std::endl;
    }
    std::cout << "Number of accepted thetas: " << naccept_theta << std::endl;
    chifile.close();
    thetafile.close();
    cudaFree(devStates);  // Free up the memory on the GPU from the RNG states
    
	return 0;
}
