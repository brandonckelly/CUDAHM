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

// Boost includes
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

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
static const int mfeat = 5;  // the number of measured features per data point
static __constant__ int c_m = 5;
    // observational frequencies, corresponding to the Herschel PACS and SPIRES bands
static const double nu[mfeat] = {4.286e12, 1.765e12, 1.200e12, 8.571e11, 6.000e11};
static __constant__ double c_nu[mfeat] = {4.286e12, 1.765e12, 1.200e12, 8.571e11, 6.000e11};
static const double hplanck = 6.6260755e-27;  // Planck's constant, in cgs
static const double clight = 2.997925e10;
static const double kboltz = 1.380658e-16;
static __constant__ double c_2h_over_csqr = 1.4745002e-47;  // 2 * hplanck / clight^2, used in the Planck function
static __constant__ double c_h_over_k = 4.7992157e-11;  // hplanck / kboltz

// Global random number generator and distributions for generating random numbers on the host. The random number generator used
// is the Mersenne Twister mt19937 from the BOOST library.
boost::random::mt19937 rng;
boost::random::normal_distribution<> snorm(0.0, 1.0); // Standard normal distribution
boost::random::uniform_real_distribution<> uniform(0.0, 1.0); // Uniform distribution from 0.0 to 1.0

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
void update_chi(double* chi, double* meas, double* meas_unc, int n, double* logdens_meas, double* logdens_pop,
		double* prop_cholfact, curandState* state, int current_iter, int* naccept, double* debug_info)
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
            // scaled_proposal_j = 0.0;
            new_chi[j] = chi[n * j + i] + scaled_proposal_j;
            scaled_proposal[j] = scaled_proposal_j;
        }

        // Compute the conditional log-posterior of the proposed parameter values for this data point
        double meas_i[mfeat];
        double meas_unc_i[mfeat];
        for (int k=0; k<c_m; k++) {
            // grab the measurements for this data point
            meas_i[k] = meas[n * k + i];
            meas_unc_i[k] = meas_unc[n * k + i];
        }

        double logdens_meas_prop = logdensity_meas(meas_i, meas_unc_i, new_chi);
        double logdens_pop_prop = logdensity_pop(new_chi, c_theta);
        double logdens_prop = logdens_meas_prop + logdens_pop_prop;

        bool finite_logdens = isfinite(logdens_prop);

        // Compute the Metropolis ratio
        double logdens_old = logdens_pop[i] + logdens_meas[i];
        double ratio = logdens_prop - logdens_old;

        if (i == 0) {
			debug_info[0] = logdens_meas_prop - logdens_meas[i];
			debug_info[1] = logdens_pop_prop - logdens_pop[i];
			debug_info[2] = ratio;
		}

        ratio = (ratio < 0.0) ? ratio : 0.0;
        ratio = exp(ratio);

        // Now randomly decide whether to accept or reject
        double unif_draw = curand_uniform_double(&localState);

        if ((unif_draw < ratio) && finite_logdens) {
            // Accept this proposal, so save this parameter value and conditional log-posterior
            for (int j=0; j<c_p; j++) {
                chi[n * j + i] = new_chi[j];
            }
            logdens_pop[i] = logdens_pop_prop;
            logdens_meas[i] = logdens_meas_prop;
            ++naccept[i];
        }

        // Copy state back to global memory
        state[i] = localState;

        if (finite_logdens) {
			// Finally, adapt the Cholesky factor of the proposal distribution
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
			for (int j=0; j<dim_cholfactor; j++) {
				prop_cholfact[n * j + i] = cholfactor[j];
			}
        }
	}
}

// calculate the logdensity of chi|meas,theta for each chi on the device, needed for computing the
// log-densities of the initial values
__global__
void g_logdens_meas(double* chi, double* meas, double* meas_unc, int ndata, double* logdens_meas)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < ndata)
	{
		double this_chi[p];
		double this_meas[mfeat];
		double this_meas_unc[mfeat];
		for (int j=0; j<p; j++) {
			// grab the characteristics for this data point
			this_chi[j] = chi[j * ndata + i];
		}
        for (int k=0; k<c_m; k++) {
            // grab the measurements for this data point
            this_meas[k] = meas[ndata * k + i];
            this_meas_unc[k] = meas_unc[ndata * k + i];
        }
        logdens_meas[i] = logdensity_meas(this_meas, this_meas_unc, this_chi);
	}
}

// calculate the logdensity of theta for each chi on the device, needed for updating theta
__global__
void g_logdens_pop(double* chi, int ndata, double* logdens_pop)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < ndata)
	{
		double this_chi[p];
		for (int j=0; j<p; j++) {
			this_chi[j] = chi[j * ndata + i];
		}
		double chi_sum = 0.0;
		for (int j = 0; j < p; ++j) {
			chi_sum += this_chi[j];
		}

		logdens_pop[i] = chi_sum;
	}
}

struct zsqr : public thrust::unary_function<double*,double> {
    __device__ __host__
    double operator()(double* chi) {
    	double chi_sum = 0.0;
    	for (int j = 0; j < p; ++j) {
			chi_sum += chi[j];
		}
        return chi_sum;
    }
};
*/

// Generate a data set and fill the chi, meas, and meas_sigma host vector with the values
void generate_data(int ndata, thrust::host_vector<double>& theta, thrust::host_vector<double>& chi, thrust::host_vector<double>& meas,
		thrust::host_vector<double>& meas_sigma) {

	double sigma_msmt[5] = {2.2e-4, 3.3e-4, 5.2e-4, 3.7e-4, 2.2e-4};  // Standard deviation for the measurement errors

	for (int j = 0; j < mfeat; ++j) {
		sigma_msmt[j] *= 1.0;
	}

	for (int i=0; i<ndata; i++) {
        // Loop over the data indices
		for (int j=0; j<p; j++) {
            // Loop over the chi indices to generate true value of the characteristics
			chi[i + ndata * j] = theta[j] + sqrt(exp(theta[p + j])) * snorm(rng);
        }
        for (int k=0; k<mfeat; k++) {
            // Loop over the feature indices to generate measurements, given the characteristics
            double bbody_numer = 2.0 * hplanck * nu[k] * nu[k] * nu[k] / (clight * clight);
            double temperature = exp(chi[ndata * 2 + i]); // Grab the temperature for this data point
            double bbody_denom = exp(hplanck * nu[k] / (kboltz * temperature)) - 1.0;
            double bbody = bbody_numer / bbody_denom;
            // Grab the SED normalization and power-law index
            double normalization = exp(chi[i]);
            double beta = chi[ndata + i];
            // Compute the model SED
            double nu_over_nu0 = nu[k] / nu0;
            double SED_ik = normalization * pow(nu_over_nu0, beta) * bbody;
            // Generate measured SEDs
			meas_sigma[k * ndata + i] = sigma_msmt[k];
            meas[k * ndata + i] = SED_ik + sigma_msmt[k] * snorm(rng);
		}
	}
}

// Initialize the chi and theta values. Right now this just initializes them to their true values. Also initialize the Cholesky factors
// for the chi and theta proposals. The elements of the chi and theta proposals are initially independent.
void initialize_parameters(int ndata, thrust::host_vector<double>& theta, thrust::host_vector<double>& chi,
		thrust::host_vector<double>& theta_cholfactor, thrust::host_vector<double>& chi_cholfactor, thrust::host_vector<double>& true_chi) {

	// For now, initialize chi values for drawing from their parent distribution.
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < p; ++j) {
			double chi_mean = theta[j];
			double chi_sig = sqrt(exp(theta[p + j]));
			chi[j * ndata + i] = chi_mean + chi_sig * snorm(rng);
		}
	}

	for (int j = 0; j < dim_theta; ++j) {
		theta[j] = theta[j] + theta[j] * 0.2 * snorm(rng);
	}

	// Initialize the cholesky factor for the theta proposals
    int diag_index = 0;
    for (int j=0; j<dim_theta; j++) {
    	// Initialize the covariance matrix of the theta proposal distribution to be 0.01 * identity(p)
        theta_cholfactor[diag_index] = 0.01;
        diag_index += j + 2;
    }

    // Initialize the cholesky factor for the chi proposals
    for (int i=0; i<ndata; i++) {
        int diag_index = 0;
        for (int j=0; j<p; j++) {
        	// Initialize the covariance matrix of the chi proposal distribution to be 0.01 * identity(p)
        	chi_cholfactor[i + ndata * diag_index] = 0.01;
        	diag_index += j + 2;
        }
    }
}

// Test the rank-1 cholesky update
void test_CholUpdateR1() {
	int dim_cf = 4;
	int size_cf = 10; //dim_cf * dim_cf - ((dim_cf - 1) * dim_cf) / 2
	double L[10] = {31.088, 7.759, 31.559, 7.632, 6.998, 29.739, 6.345, 7.080, 6.108, 29.015};
	// true values of the updated cholesky factor for A = L * transpose(L) +/- v * transpose(v)
	double Lup0[10] = {31.542, 4.726, 36.504, 6.898, 8.527, 29.748, 7.414, 3.262, 5.894, 30.270};
	double Ldown0[10] = {30.628, 10.884, 25.278, 8.389, 4.944, 29.718, 5.246, 13.219, 6.581, 26.017};
	double v[4] = {-5.331, 17.284, 3.691, -6.861};

	double Ldown[10];
	double Lup[10];
	double vup[4];
	double vdown[4];
	for (int i=0; i<size_cf; i++) {
		Lup[i] = L[i];
		Ldown[i] = L[i];
	}
	for (int i=0; i<dim_cf; i++) {
		vup[i] = v[i];
		vdown[i] = v[i];
	}

	// test the update first
	CholUpdateR1(Lup, vup, dim_cf, false);
	CholUpdateR1(Ldown, vdown, dim_cf, true);
	double fracdiff_up = 0.0;
	double fracdiff_down = 0.0;
	for (int i=0; i<size_cf; i++) {
		std::cout << "Lup[i]: " << Lup[i] << ", Ldown[i]: " << Ldown[i] << std::endl;
		double fracdiff = abs(Lup[i] - Lup0[i]) / Lup0[i];
		fracdiff_up = max(fracdiff_up, fracdiff);
		fracdiff = abs(Ldown[i] - Ldown0[i]) / Ldown0[i];
		fracdiff_down = max(fracdiff_down, fracdiff);
	}
	std::cout << "test_CholUpdateR1:" << std::endl;
	std::cout << "Maximum fractional difference for rank-1 update: " << fracdiff_up << std::endl;
	std::cout << "Maximum fractional difference for rank-1 downdat: " << fracdiff_down << std::endl;
}

int main(void)
{
    /* TODO: Add command line options */

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

    int ndata = 1000; // # of data points

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

    // Allocate memory for arrays of chi, measurement, and measurement uncertainties on host
    thrust::host_vector<double> h_meas(ndata * mfeat);  // The measurements, mfeat values for each of ndata data points
	thrust::host_vector<double> h_meas_unc(ndata * mfeat);  // The measurement uncertainties
    thrust::host_vector<double> h_chi(ndata * p);  // Unknown characteristics, p values for each of ndata data points
    thrust::host_vector<double> h_true_chi(ndata * p);

    // Generate characteristics and data
    std::cout << "Generating some data..." << std::endl;
    generate_data(ndata, h_theta, h_true_chi, h_meas, h_meas_unc);

    // Allocate memory for cholesky factors on host.
    //
    // Cholesky factor of chi proposal distribution for each data point, used by the Metropolis algorithm. The proposal
    // covariance matrix of chi for each data point is Covar_i = PropChol_i * transpose(PropChol_i).
    thrust::host_vector<double> h_chi_cholfactor(ndata * dim_cholfactor);
    thrust::fill(h_chi_cholfactor.begin(), h_chi_cholfactor.end(), 0.0);

    // Cholesky factor for the covariance matrix of the theta proposal distribution
    int dim_cholfactor_theta = dim_theta * dim_theta - ((dim_theta - 1) * dim_theta) / 2;
    thrust::host_vector<double> h_theta_cholfact(dim_cholfactor_theta);

    // Initialize the chi values, as well as the cholesky factors for the theta and chi proposals
    initialize_parameters(ndata, h_theta, h_chi, h_theta_cholfact, h_chi_cholfactor, h_true_chi);

    // Allocate memory for arrays on device and copy the values from the host
    thrust::device_vector<double> d_meas = h_meas;
    thrust::device_vector<double> d_meas_unc = h_meas_unc;
    thrust::device_vector<double> d_chi = h_chi;
    thrust::device_vector<double> d_chi_cholfactor = h_chi_cholfactor;
    thrust::device_vector<double> d_logdens_meas(ndata);  // log densities of chi values
    thrust::device_vector<double> d_logdens_pop(ndata); // log densities of theta values for each chi

   	// Load the initial guess for theta into constant memory
    double* p_theta = thrust::raw_pointer_cast(&h_theta[0]);
	CUDA_CALL(cudaMemcpyToSymbol(c_theta, p_theta, h_theta.size() * sizeof(*p_theta)));

    // Cuda grid launch
    dim3 nThreads(256);
    dim3 nBlocks((ndata + nThreads.x-1) / nThreads.x);
    printf("nBlocks: %d\n", nBlocks.x);  // no more than 64k blocks!
    if (nBlocks.x > 65535)
    {
        std::cerr << "ERROR: Block is too large" << std::endl;
        return 2;
    }

    /*******************************
    	Now run the MCMC sampler
	*******************************/
    // Set up parallel random number generators on the GPU
    curandState* devStates;  // Create state object for random number generators on the GPU
    // Allocate memory on GPU for RNG states
    CUDA_CALL(cudaMalloc((void **)&devStates, nThreads.x * nBlocks.x *
    		sizeof(curandState)));
     // Initialize the random number generator states on the GPU
     setup_kernel<<<nBlocks,nThreads>>>(devStates);

     // Wait until RNG stuff is done running on the GPU, make sure everything went OK
     CUDA_CALL(cudaDeviceSynchronize());

    //std::ofstream chifile("chis.dat");
    std::ofstream thetafile("thetas.dat");
    std::ofstream chifile("chi0.dat");
    int mcmc_iter = 50000;
    int naccept_theta = 0;
    std::cout << "Running MCMC Sampler...." << std::endl;

    thrust::host_vector<double> h_logdens_meas;
    thrust::host_vector<double> h_logdens_pop;
    thrust::host_vector<int> h_naccept(ndata);
    thrust::fill(h_naccept.begin(), h_naccept.end(), 0);
    thrust::device_vector<int> d_naccept = h_naccept;

    thrust::device_vector<double> d_debug_info(3);
    thrust::host_vector<double> h_debug_info = d_debug_info;

    // Initialize the log-posteriors
    double* p_meas = thrust::raw_pointer_cast(&d_meas[0]); // CUDA kernels need to have the raw pointers
    double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
    double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
    double* p_logdens_meas = thrust::raw_pointer_cast(&d_logdens_meas[0]);
    double* p_logdens_pop = thrust::raw_pointer_cast(&d_logdens_pop[0]);
    g_logdens_pop<<<nBlocks,nThreads>>>(p_chi, ndata, p_logdens_pop);
    g_logdens_meas<<<nBlocks,nThreads>>>(p_chi, p_meas, p_meas_unc, ndata, p_logdens_meas);
    CUDA_CALL(cudaDeviceSynchronize());

    for (int i=0; i<mcmc_iter; i++) {
        // Now grab the pointers to the vectors, needed to run the kernel since it doesn't understand Thrust
        // We do this here because the thrust vectors are smart, and we want to make sure they don't reassign
        // memory for whatever reason. This is very cheap to do, so better safe than sorry.
        double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
        double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
        double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
        double* p_prop_cholfact = thrust::raw_pointer_cast(&d_chi_cholfactor[0]);
        double* p_logdens_meas = thrust::raw_pointer_cast(&d_logdens_meas[0]);
        double* p_logdens_pop = thrust::raw_pointer_cast(&d_logdens_pop[0]);
        int* p_naccept = thrust::raw_pointer_cast(&d_naccept[0]);
        int current_iter = i + 1;

        /*
        std::cout << "current h_chi: " << std::endl;
        for (int j = 0; j < p; ++j) {
        	std::cout << h_chi[j*ndata] << " ";
		}
        std::cout << std::endl;
		*/

        // Update the characteristics on the GPU
        double* p_debug_info = thrust::raw_pointer_cast(&d_debug_info[0]);
        update_chi<<<nBlocks,nThreads>>>(p_chi, p_meas, p_meas_unc, ndata, p_logdens_meas, p_logdens_pop,
        		p_prop_cholfact, devStates, current_iter, p_naccept, p_debug_info);

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
            //scaled_proposal[j] = 0.0;
            proposed_theta[j] = h_theta[j] + scaled_proposal[j];
        }

        CUDA_CALL(cudaDeviceSynchronize());
        h_chi = d_chi;
        h_logdens_pop = d_logdens_pop;
        h_logdens_meas = d_logdens_meas;

        /*
        std::cout << "new h_chi: " << std::endl;
        for (int j = 0; j < p; ++j) {
        	std::cout << h_chi[j*ndata] << " ";
		}
        std::cout << std::endl;

        h_debug_info = d_debug_info;
        std::cout << "Difference in logdens_pop: " << h_debug_info[0] << std::endl;
        std::cout << "Difference in logdens_meas: " << h_debug_info[1] << std::endl;
        std::cout << "Difference in logdens: " << h_debug_info[2] << std::endl;
		*/

        // Update the theta values

        // get log-density of current value of theta
        double logdens_old = thrust::reduce(d_logdens_pop.begin(), d_logdens_pop.end());

        // get value of population log-density for proposed theta value for each chi value. note that
        // this overwrites the value of d_logdens_pop.
        double* p_theta = thrust::raw_pointer_cast(&proposed_theta[0]);
        // copy the proposed theta to constant memory on the device
        CUDA_CALL(cudaMemcpyToSymbol(c_theta, p_theta, proposed_theta.size() * sizeof(*p_theta)));
        p_chi = thrust::raw_pointer_cast(&d_chi[0]);  // grab the pointers in case thrust changed the location
        p_logdens_pop = thrust::raw_pointer_cast(&d_logdens_pop[0]);

        g_logdens_pop<<<nBlocks,nThreads>>>(p_chi, ndata, p_logdens_pop);
        CUDA_CALL(cudaDeviceSynchronize());

        double logdens_prop = thrust::reduce(d_logdens_pop.begin(), d_logdens_pop.end());

        double lograt = logdens_prop - logdens_old;
        lograt = std::min(lograt, 0.0);
        double ratio = exp(lograt);
        double unif = uniform(rng);

        if (unif < ratio) {
            // Accept the proposed theta
            h_theta = proposed_theta;
            h_logdens_pop = d_logdens_pop;
            naccept_theta++;
        } else {
			// keep current value of theta, but need to restore the values of d_logdens_pop overwritten by
        	// g_logdens_pop and move the current value of theta back to constant memory on the device
        	d_logdens_pop = h_logdens_pop;
            p_theta = thrust::raw_pointer_cast(&h_theta[0]);
            // copy the proposed theta to constant memory on the device
            CUDA_CALL(cudaMemcpyToSymbol(c_theta, p_theta, h_theta.size() * sizeof(*p_theta)));
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
        double logdens_pop = thrust::reduce(d_logdens_pop.begin(), d_logdens_pop.end());
        thetafile << logdens_pop << std::endl;
        // Save the chi values
        for (int j=0; j<p; j++) {
        	chifile << h_chi[ndata * j] << " ";
        }
        double logdens_meas = thrust::reduce(d_logdens_meas.begin(), d_logdens_meas.end());
        chifile << logdens_meas << std::endl;
    }
    thetafile.close();

    // Print out information on acceptance rates
    std::cout << "Number of accepted thetas: " << naccept_theta << std::endl;
    int max_chiaccept = 0;
    int min_chiaccept = mcmc_iter;
    double mean_chiaccept = 0.0;
    h_naccept = d_naccept;
    for (int i=0; i<ndata; i++) {
    	min_chiaccept = min(min_chiaccept, h_naccept[i]);
    	max_chiaccept = max(max_chiaccept, h_naccept[i]);
    	mean_chiaccept += h_naccept[i];
    }
    mean_chiaccept = mean_chiaccept / ndata;
    std::cout << "Minimum number of accepted chi values: " << min_chiaccept << std::endl;
    std::cout << "Mean number of accepted chi values: " << mean_chiaccept << std::endl;
    std::cout << "Maximum number of accepted chi values: " << max_chiaccept << std::endl;

    cudaFree(devStates);  // Free up the memory on the GPU from the RNG states

    std::cout << "True chi[0] values: ";
    for (int j=0; j<p; j++) {
    	std::cout << h_true_chi[j * ndata] << " ";
    }
    std::cout << std::endl;

    std::cout << "Measured SED[0] values (nu, flux, error): ";
    for (int k=0; k<mfeat; k++) {
    	std::cout << nu[k] << " " << h_meas[k * ndata] << " " << h_meas_unc[k * ndata] << std::endl;
    }

	return 0;
}



