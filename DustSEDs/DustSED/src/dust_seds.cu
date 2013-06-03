/*
 * dust_seds.cu
 *
 *  Created on: Jun 2, 2013
 *      Author: brandonkelly
 */

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
#include <iostream>
#include <math.h>
#include <vector>

//#include <thrust/sort.h>
//#include <thrust/copy.h>

//#include <algorithm>
//#include <cstdlib>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)

/****************** CONSTANTS ******************/

// dimension of the characteristics, chi
static const int p = 3;
static __constant__ int c_p = 3;
static const int dim_cholfactor = 6; // = p * p - ((p - 1) * p) / 2;
static const int c_dim_cholfactor = 6;

// dimension of the chi population parameter, theta
static const int dim_theta = 6;
static __constant__ int c_dim_theta = 6;

// Target acceptance rate for Robust Adaptive Metropolis (RAM).
static __constant__ double c_target_rate = 0.4;
static const double theta_target_rate = 0.4;

// Decay rate for proposal scaling factor updates. The proposal scaling factors decay as 1 / niter^decay_rate.
// This is gamma in the notation of Vihola (2012)
static __constant__ double c_decay_rate = 0.66667;
static const double decay_rate = 0.666667;

// Pointer to the current value of theta. This is stored in constant memory so that all the threads on the GPU
// can access the same theta quickly.
static __constant__ double c_theta[dim_theta];

// Constants used for the model SED
static __constant__ double c_nu0 = 2.3e11;  // reference frequency, nu0 = 230 GHz
static const double nu0 = 2.3e11;
static const int m = 5;  // the number of measured features per data point
static __constant__ int c_m = 5;
    // observational frequencies, corresponding to the Herschel PACS and SPIRES bands
static const double nu[m] = {4.286e12, 1.765e12, 1.200e12, 8.571e11, 6.000e11};
static __constant__ double c_nu[m] = {4.286e12, 1.765e12, 1.200e12, 8.571e11, 6.000e11};
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

// Function to compute the conditional log-density of the measurements given the chi value
__device__
double logdensity_meas(double* meas, double* meas_unc, double* chi) {

    double normalization, beta, temperature;
    normalization = exp(chi[0]);
    beta = chi[1];
    temperature = exp(chi[2]);

    double logdensity = 0.0;
    for (int k=0; k<c_m; k++) {
        double model_sed = modified_bbody(c_nu[k], normalization, beta, temperature);
        double stnd_meas = (meas[k] - model_sed) / meas_unc[k];
        logdensity += -0.5 * stnd_meas * stnd_meas;
    }
    return logdensity;
}

// Function to compute the conditional log-density of the chi values given theta
__device__ __host__
double logdensity_pop(double* chi, double* theta) {
#ifdef __CUDA_ARCH__
    int dim_chi = c_p;  // function is called from device, so need to access constant memory on the GPU
    int ntheta = c_dim_theta;
#else
    int dim_chi = p;
    int ntheta = dim_theta;
#endif
    // right now this is just an independent p-dimensional gaussian distribution, for simplicity
    double logdensity = 0.0;
    for (int j=0; j<dim_chi; j++) {
        double mu_j = theta[j];
        double var_j = exp(theta[j + dim_chi]);
        double centered_chi_j = chi[j] - mu_j;
        logdensity += -0.5 * log(var_j) - 0.5 * centered_chi_j * centered_chi_j / var_j;
    }
    return logdensity;
}

// Function to compute the rank-1 Cholesky update/downdate. Note that this is done in place.
__device__ __host__
void CholUpdateR1(double* cholfactor, double* v, int dim_v, bool downdate) {

    double sign = 1.0;
	if (downdate) {
		// Perform the downdate instead
		sign = -1.0;
	}
    int diag_index = 0;  // index of the diagonal of the cholesky factor
	for (int i=0; i<dim_v; i++) {
        // loop over the columns of the Cholesky factor
        double L_ii = cholfactor[diag_index];
        double v_i = v[i];
        double r = sqrt( L_ii * L_ii + sign * v_i * v_i);
		double c = r / L_ii;
		double s = v_i / L_ii;
		cholfactor[diag_index] = r;
        int index_ji = diag_index; // index of the cholesky factor array that points to L[j,i]
        // update the rest of the rows of the Cholesky factor for this column
        for (int j=i+1; j<dim_v; j++) {
            // loop over the rows of the i^th column of the Cholesky factor
            index_ji += j;
            cholfactor[index_ji] = (cholfactor[index_ji] + sign * s * v[j]) / c;
        }
        // update the elements of the vector v[i+1:dim_v-1]
        index_ji = diag_index;
        for (int j=i+1; j<dim_v; j++) {
            index_ji += j;
            v[j] = c * v[j] - s * cholfactor[index_ji];
        }
        diag_index += i + 2;
    }
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

// Perform the MHA update on the n characteristics, done in parallel on the GPU.
__global__
void update_chi(double* chi, double* meas, double* meas_unc, int n, double* logdens, double* prop_cholfact,
                curandState* state, int current_iter)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n)
	{
        // Copy variables to local memory for efficiency
        curandState localState = state[i];
        double cholfactor[dim_cholfactor];
        for (int j=0; j<dim_cholfactor; j++) {
            cholfactor[j] = prop_cholfact[n * j + i];
        }

        // get the unit proposal
        double snorm_deviate[p];
        for (int j=0; j<c_p; j++) {
            snorm_deviate[j] = curand_normal_double(&localState);
        }

        // propose a new chi value
        double new_chi[p];
        double scaled_proposal[p];
        int cholfact_index = 0;
        for (int j=0; j<c_p; j++) {
            double scaled_proposal_j = 0.0;
            for (int k=0; k<(j+1); k++) {
                // transform the unit proposal to the centered proposal, drawn from a multivariate normal
                // or t-distribution.
                scaled_proposal_j += cholfactor[cholfact_index] * snorm_deviate[k];
                cholfact_index++;
            }
            scaled_proposal_j = 0.0;
            new_chi[j] = chi[n * j + i] + scaled_proposal_j;
            scaled_proposal[j] = scaled_proposal_j;
        }

        // Compute the conditional log-posterior of the proposed parameter values for this data point
        double meas_i[m];
        double meas_unc_i[m];
        for (int k=0; k<c_m; k++) {
            // grab the measurements for this data point
            meas_i[k] = meas[n * k + i];
            meas_unc_i[k] = meas_unc[n * k + i];
        }

        double logdens_meas = logdensity_meas(meas_i, meas_unc_i, new_chi);
        double logdens_pop = logdensity_pop(new_chi, c_theta);
        double logdens_prop = logdens_meas + logdens_pop;

        // Compute the Metropolis ratio
        double ratio = logdens_prop - logdens[i] - logdens_meas;
        ratio = (ratio < 0.0) ? ratio : 0.0;
        ratio = exp(ratio);

        // Now randomly decide whether to accept or reject
        double unif_draw = curand_uniform_double(&localState);

        if (unif_draw < ratio) {
            // Accept this proposal, so save this parameter value and conditional log-posterior
            for (int j=0; j<c_p; j++) {
                chi[n * j + i] = new_chi[j];
            }
            logdens[i] = logdens_pop;
        }

        // Copy state back to global memory
        state[i] = localState;

        // Finally, adapt the Cholesky factor of the proposal distribution
        // TODO: check that the ratio is finite before doing this step
        double unit_norm = 0.0;
        for (int j=0; j<c_p; j++) {
            unit_norm += snorm_deviate[j] * snorm_deviate[j];
        }
        unit_norm = sqrt(unit_norm);
        double decay_sequence = 1.0 / pow((double) current_iter, c_decay_rate);
        double scaled_coef = sqrt(decay_sequence * fabs(ratio - c_target_rate)) / unit_norm;
        for (int j=0; j<c_p; j++) {
            scaled_proposal[j] *= scaled_coef;
        }
        bool downdate = (ratio < c_target_rate);
        CholUpdateR1(cholfactor, scaled_proposal, c_p, downdate);

        // copy cholesky factor for this data point back to global memory
        cholfact_index = 0;
        for (int j=0; j<dim_cholfactor; j++) {
            prop_cholfact[n * j + i] = cholfactor[j];
        }
	}
}

struct zsqr : public thrust::unary_function<double,double> {
    double mu, var;
    zsqr(double m, double v) : mu(m), var(v) {}

    __device__ __host__
    double operator()(double chi) {
        double chi_cent = chi - mu;
        double logdens_pop = -0.5 * log(var) - 0.5 * chi_cent * chi_cent / var;
        return logdens_pop;
    }
};


int main(void)
{
    /* TODO: Should use the boost random number generator libraries for doing this on the host */

    // Random number generator and distribution needed to simulate some data
	// unsigned int seed = 98724732;
    unsigned int seed = 123455;
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

    int n = 3; // # of data points

    double mu_norm = 8.5 * log(10.0);  // Average value of the natural logarithm of the SED normalization
    double sig_norm = 0.5 * log(10.0);  // Standard deviation in the SED normalization
    double mu_beta = 2.0;  // Average value of the SED power-law index
    double sig_beta = 0.2;  // Standard deviation in the SED power-law index
    double mu_temp = log(15.0);  // Average value of the logarithm of the dust temperature
    double sig_temp = 0.2;  // Standard deviation in the logarithm of the SED temperature

    thrust::host_vector<double> h_theta(dim_theta);  // Allocate memory on host
    h_theta[0] = mu_norm;
    h_theta[1] = mu_beta;
    h_theta[2] = mu_temp;
    h_theta[3] = log(sig_norm * sig_norm);
    h_theta[4] = log(sig_beta * sig_beta);
    h_theta[5] = log(sig_temp * sig_temp);

    // Construct initial cholesky factor for the covariance matrix of the theta proposal distribution
    int dim_cholfactor_theta = dim_theta * dim_theta - ((dim_theta - 1) * dim_theta) / 2;
    thrust::host_vector<double> h_theta_cholfact(dim_cholfactor_theta);
    int diag_index = 0;
    for (int j=0; j<dim_theta; j++) {
        h_theta_cholfact[diag_index] = 1e-2;
        diag_index += j + 2;
    }

    // Allocate memory for arrays on host
    thrust::host_vector<double> h_meas(n * m);  // The measurements, m values for each of n data points
	thrust::host_vector<double> h_meas_unc(n * m);  // The measurement uncertainties
    thrust::host_vector<double> h_chi(n * p);  // Unknown characteristics, p values for each of n data points

    // Cholesky factor of proposal distribution for each data point, used by the Metropolis algorithm. The proposal
    // covariance matrix for each data point is Covar_i = PropChol_i * transpose(PropChol_i).
    thrust::host_vector<double> h_prop_cholfact(n * dim_cholfactor);
    thrust::fill(h_prop_cholfact.begin(), h_prop_cholfact.end(), 0.0);

	// Create simulated characteristics and data
    thrust::host_vector<double> h_true_chi(n * p);
    double sigma_msmt[5] = {2.2e-4, 3.3e-4, 5.2e-4, 3.7e-4, 2.2e-4};  // Standard deviation for the measurement errors

    std::cout << "Generating some data..." << std::endl;

	for (int i=0; i<n; i++) {
        // Loop over the data indices
        int diag_index = 0;
		for (int j=0; j<p; j++) {
            // Loop over the chi indices to generate true value of the characteristics
			h_true_chi[i + n * j] = h_theta[j] + sqrt(exp(h_theta[p + j])) * snorm(rng);
            // Initialize the covariance matrix of the chi proposal distribution to be 0.01 * identity(p)
            h_prop_cholfact[i + n * diag_index] = 0.01;
            diag_index += j + 2;
        }
        for (int k=0; k<m; k++) {
            // Loop over the feature indices to generate measurements, given the characteristics
            double bbody_numer = 2.0 * hplanck * nu[k] * nu[k] * nu[k] / (clight * clight);
            double temperature = exp(h_true_chi[n * 2 + i]); // Grab the temperature for this data point
            double bbody_denom = exp(hplanck * nu[k] / (kboltz * temperature)) - 1.0;
            double bbody = bbody_numer / bbody_denom;
            // Grab the SED normalization and power-law index
            double normalization = exp(h_true_chi[i]);
            double beta = h_true_chi[n + i];
            // Compute the model SED
            double nu_over_nu0 = nu[k] / nu0;
            double SED_ik = normalization * pow(nu_over_nu0, beta) * bbody;
            // Generate measured SEDs
			h_meas_unc[k * n + i] = sigma_msmt[k];
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
    double* p_theta = &h_theta[0];
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

    //std::ofstream chifile("chis.dat");
    std::ofstream thetafile("thetas.dat");

    int mcmc_iter = 2;
    int naccept_theta = 0;
    std::cout << "Running MCMC Sampler...." << std::endl;

    thrust::host_vector<double> h_logdens;

    for (int i=0; i<mcmc_iter; i++) {
        // Now grab the pointers to the vectors, needed to run the kernel since it doesn't understand Thrust
        // We do this here because the thrust vector are smart, and we want to make sure they don't reassign
        // memory for whatever reason. This is very cheap to do, so better safe than sorry.
        double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
        double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
        double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
        double* p_prop_cholfact = thrust::raw_pointer_cast(&d_prop_cholfact[0]);
        double* p_logdens = thrust::raw_pointer_cast(&d_logdens[0]);
        int current_iter = i + 1;
        update_chi<<<nBlocks,nThreads>>>(p_chi, p_meas, p_meas_unc, n, p_logdens, p_prop_cholfact, devStates, current_iter);

        // Generate new theta in parallel with GPU calculation above
        thrust::host_vector<double> proposed_theta(dim_theta);

            // get the unit proposal
        thrust::host_vector<double> snorm_deviate(dim_theta);
        for (int j=0; j<dim_theta; j++) {
            snorm_deviate[j] = snorm(rng);
        }

            // transform unit proposal so that is has a multivariate normal distribution
        thrust::host_vector<double> scaled_proposal(dim_theta);
        thrust::fill(scaled_proposal.begin(), scaled_proposal.end(), 0.0);
        int cholfact_index = 0;
        for (int j=0; j<dim_theta; j++) {
            for (int k=0; k<(j+1); k++) {
                scaled_proposal[j] += h_theta_cholfact[cholfact_index] * snorm_deviate[k];
                cholfact_index++;
            }
            scaled_proposal[j] = 0.0;
            proposed_theta[j] = h_theta[j] + scaled_proposal[j];
        }

        CUDA_CALL(cudaDeviceSynchronize());
        h_chi = d_chi;
        h_logdens = d_logdens;

        // Compute Metropolis ratio
        //
        // right now we loop over the elements of chi because we assume that they are statistically independent, i.e.,
        // the theta array does not contain correlations among the element of chi. this is unrealistic, and we will need
        // to come up with a better way to do the transform + reduction: maybe use CUBLAS?
        thrust::device_vector<double>::iterator chi_iter_begin = d_chi.begin();

        double logdens_pop = 0.0;
        for (int j=0; j<dim_theta/2; j++) {
            double proposed_mu_j = proposed_theta[j];
            double proposed_var_j = exp(proposed_theta[j + p]);
            // transform and reduction is over d_chi[j * n : (j+1) * n - 1]
            zsqr zsqrj(proposed_mu_j, proposed_var_j);
            logdens_pop = thrust::transform_reduce(chi_iter_begin, chi_iter_begin+n,
                                                zsqrj, logdens_pop, thrust::plus<double>());
            thrust::advance(chi_iter_begin, n);
        }

        double logdens_old = thrust::reduce(d_logdens.begin(), d_logdens.end());
        double lograt = logdens_pop - logdens_old;
        lograt = std::min(lograt, 0.0);
        double ratio = exp(lograt);
        double unif = uniform(rng);

        if (unif < ratio) {
            // Accept the proposed theta
            h_theta = proposed_theta;
            // grab the pointer in case thrust changed the memory location on us
            double* p_theta = thrust::raw_pointer_cast(&h_theta[0]);
            // copy the new theta to constant memory on the device
            CUDA_CALL(cudaMemcpyToSymbol(c_theta, p_theta, h_theta.size() * sizeof(*p_theta)));
            naccept_theta++;
        }

        // finally, adapt the proposal covariance of theta by updating its cholesky factor
        double unit_norm = 0.0;
        for (int j=0; j<dim_theta; j++) {
            unit_norm += snorm_deviate[j] * snorm_deviate[j];
        }

        unit_norm = sqrt(unit_norm);
        double decay_sequence = 1.0 / pow(current_iter, decay_rate);
        double scaled_coef = sqrt(decay_sequence * fabs(ratio - theta_target_rate)) / unit_norm;
        for (int j=0; j<dim_theta; j++) {
            scaled_proposal[j] *= scaled_coef;
        }

        bool downdate = (ratio < theta_target_rate);
        double* p_theta_cholfact = thrust::raw_pointer_cast(&h_theta_cholfact[0]);
        double* p_scaled_proposal = thrust::raw_pointer_cast(&scaled_proposal[0]);
        CholUpdateR1(p_theta_cholfact, p_scaled_proposal, dim_theta, downdate);

        // Save the theta values
        for (int j=0; j<dim_theta; j++) {
            thetafile << h_theta[j] << " ";
        }
        thetafile << std::endl;
    }
    std::cout << "Number of accepted thetas: " << naccept_theta << std::endl;
    thetafile.close();

    cudaFree(devStates);  // Free up the memory on the GPU from the RNG states

	return 0;
}



