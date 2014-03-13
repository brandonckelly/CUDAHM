/*
 * kernels.cuh
 *
 *  Created on: Jul 2, 2013
 *      Author: brandonkelly
 */

#ifndef KERNELS_H__
#define KERNELS_H__

// Cuda Includes
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Boost includes
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

// pointers to functions that must be supplied by the user for computing the conditional log-densities
typedef double (*pLogDensMeas)(double*, double*, double*, int, int);
typedef double (*pLogDensPop)(double*, double*, int, int);

// Global random number generator and distributions for generating random numbers on the host. The random number generator used
// is the Mersenne Twister mt19937 from the BOOST library.
extern boost::random::mt19937 rng;
extern boost::random::normal_distribution<> snorm; // Standard normal distribution
extern boost::random::uniform_real_distribution<> uniform; // Uniform distribution from 0.0 to 1.0

// Place population parameter array in constant memory since all threads access the same values. If more than 100 elements are
// needed, just change this here and in kernels.cu.
__constant__ extern double c_theta[100];

/*
 * FUNCTION DEFINITIONS
 */

// Function to compute the rank-1 Cholesky update/downdate. Note that this is done in place.
__device__ __host__
void chol_update_r1(double* cholfactor, double* v, int dim_v, bool downdate) {

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

// propose a new value for the characteristic
__device__ __host__
void Propose(double* chi, double* cholfact, double* proposed_chi, double* snorm_deviate,
		double* scaled_proposal, int pchi, curandState* p_state)
{
	// get the unit proposal
	for (int j=0; j<pchi; j++) {
#ifdef __CUDA_ARCH__
		snorm_deviate[j] = curand_normal_double(p_state);
#else
		snorm_deviate[j] = snorm(rng);
#endif
	}

	// propose a new chi value
	int cholfact_index = 0;
	for (int j=0; j<pchi; j++) {
		double scaled_proposal_j = 0.0;
		for (int k=0; k<(j+1); k++) {
			// transform the unit proposal to the centered proposal, drawn from a multivariate normal.
			scaled_proposal_j += cholfact[cholfact_index] * snorm_deviate[k];
			cholfact_index++;
		}
		proposed_chi[j] = chi[j] + scaled_proposal_j;
		scaled_proposal[j] = scaled_proposal_j;
	}
}

// adapt the covariance matrix of the proposals for the characteristics
__device__ __host__
void AdaptProp(double* cholfact, double* snorm_deviate, double* scaled_proposal, double metro_ratio,
		int pchi, int current_iter)
{
	double unit_norm = 0.0;
	for (int j=0; j<pchi; j++) {
		unit_norm += snorm_deviate[j] * snorm_deviate[j];
	}
	unit_norm = sqrt(unit_norm);
	double decay_rate = 0.667;
	double target_rate = 0.4;
	double decay_sequence = 1.0 / pow(current_iter, decay_rate);
	double scaled_coef = sqrt(decay_sequence * fabs(metro_ratio - target_rate)) / unit_norm;
	for (int j=0; j<pchi; j++) {
		scaled_proposal[j] *= scaled_coef;
	}
	bool downdate = (metro_ratio < target_rate);
	// do rank-1 cholesky update to update the proposal covariance matrix
	chol_update_r1(cholfact, scaled_proposal, pchi, downdate);
}

// decide whether to accept or reject the proposal based on the metropolist-hasting ratio
__device__ __host__
bool AcceptProp(double logdens_prop, double logdens_current, double forward_dens,
		double backward_dens, double& ratio, curandState* p_state)
{
	double lograt = logdens_prop - forward_dens - (logdens_current - backward_dens);
	lograt = min(lograt, 0.0);
	ratio = exp(lograt);
#ifdef __CUDA_ARCH__
	double unif = curand_uniform_double(p_state);
#else
	double unif = uniform(rng);
#endif
	bool accept = (unif < ratio) && isfinite(ratio);
	return accept;
}

/*
 * KERNELS
 */

// Initialize the parallel random number generator state on the device
__global__ void initialize_rng(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
     number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

// calculate initial value of characteristics
template<int mfeat, int pchi> __global__
void initial_chi_value(double* chi, double* meas, double* meas_unc, double* cholfact, double* logdens,
		int ndata, pLogDensMeas LogDensityMeas)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		for (int j = 0; j < pchi; ++j) {
			chi[idata + j * ndata] = 0.0; // initialize chi values to zero
		}

		// set initial covariance matrix of the chi proposals as the identity matrix
		int diag_index = 0;
		for (int j=0; j<pchi; j++) {
			cholfact[idata + ndata * diag_index] = 1.0;
			diag_index += j + 2;
		}
		// copy value to registers
		double this_chi[3];
		for (int j = 0; j < pchi; ++j) {
			this_chi[j] = chi[j * ndata + idata];
		}
		double local_meas[3], local_meas_unc[3];
		for (int j = 0; j < mfeat; ++j) {
			local_meas[j] = meas[j * ndata + idata];
			local_meas_unc[j] = meas_unc[j * ndata + idata];
		}
		logdens[idata] = LogDensityMeas(this_chi, local_meas, local_meas_unc, pchi, mfeat);
	}
}

// kernel to update the values of the characteristics in parallel on the GPU
template<int mfeat, int pchi, int dtheta> __global__
void update_characteristic(double* meas, double* meas_unc, double* chi, double* cholfact,
		double* logdens_meas, double* logdens_pop, curandState* devStates, pLogDensMeas LogDensityMeas,
		pLogDensPop LogDensityPop, int current_iter, int* naccept, int ndata)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		curandState localState = devStates[idata]; // grab state of this random number generator

		// copy values for this data point to registers for speed
		// TODO: convert these arrays to shared memory
		double snorm_deviate[pchi], scaled_proposal[pchi], proposed_chi[pchi], local_chi[pchi];
		const int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
		double local_cholfact[dim_cholfact];
		int cholfact_index = 0;
		for (int j = 0; j < pchi; ++j) {
			local_chi[j] = chi[j * ndata + idata];
			for (int k = 0; k < (j+1); ++k) {
				local_cholfact[cholfact_index] = cholfact[cholfact_index * ndata + idata];
				cholfact_index++;
			}
		}
		double local_meas[mfeat], local_meas_unc[mfeat];
		for (int j = 0; j < mfeat; ++j) {
			local_meas[j] = meas[j * ndata + idata];
			local_meas_unc[j] = meas_unc[j * ndata + idata];
		}
		// propose a new value of chi
		Propose(local_chi, local_cholfact, proposed_chi, snorm_deviate, scaled_proposal, pchi, &localState);

		// get value of log-posterior for proposed chi value
		double logdens_meas_prop = LogDensityMeas(proposed_chi, local_meas, local_meas_unc, mfeat, pchi);
		double logdens_pop_prop = LogDensityPop(proposed_chi, c_theta, pchi, dtheta);
		double logpost_prop = logdens_meas_prop + logdens_pop_prop;

//		if (idata == 0) {
//			printf("current iter, idata, mfeat, pchi, dtheta: %i, %i, %i, %i, %i\n", current_iter, idata, mfeat, pchi, dtheta);
//			printf("  measurements: %g, %g, %g\n", local_meas[0], local_meas[1], local_meas[2]);
//			printf("  measurement sigmas: %g, %g, %g\n", local_meas_unc[0], local_meas_unc[1], local_meas_unc[2]);
//			printf("  cholfact: %g, %g, %g, %g, %g, %g\n", local_cholfact[0], local_cholfact[1], local_cholfact[2], local_cholfact[3],
//					local_cholfact[4], local_cholfact[5]);
//			printf("  current chi: %g, %g, %g\n", local_chi[0], local_chi[1], local_chi[2]);
//			printf("  proposed chi: %g, %g, %g\n", proposed_chi[0], proposed_chi[1], proposed_chi[2]);
//			printf("  current logdens_meas, logdens_pop: %g, %g\n", logdens_meas[idata], logdens_pop[idata]);
//			printf("  proposed logdens_meas, logdens_pop: %g, %g\n", logdens_meas_prop, logdens_pop_prop);
//			printf("\n");
//		}

		// accept the proposed value of the characteristic?
		double logpost_current = logdens_meas[idata] + logdens_pop[idata];
		double metro_ratio = 0.0;
		bool accept = AcceptProp(logpost_prop, logpost_current, 0.0, 0.0, metro_ratio, &localState);

		// adapt the covariance matrix of the characteristic proposal distribution
		AdaptProp(local_cholfact, snorm_deviate, scaled_proposal, metro_ratio, pchi, current_iter);

		for (int j = 0; j < dim_cholfact; ++j) {
			// copy value of this adapted cholesky factor back to global memory
			cholfact[j * ndata + idata] = local_cholfact[j];
		}

		// copy local RNG state back to global memory
		devStates[idata] = localState;

		// TODO: try to avoid branching statement
		// printf("current iter, Accept, idata: %d, %d, %d\n", current_iter, accept, idata);
		if (accept) {
			// accepted this proposal, so save new value of chi and log-densities
			for (int j=0; j<pchi; j++) {
				chi[ndata * j + idata] = proposed_chi[j];
			}
			logdens_meas[idata] = logdens_meas_prop;
			logdens_pop[idata] = logdens_pop_prop;
			naccept[idata] += 1;
		}
	}

}

// compute the conditional log-posterior density of the characteristics given the population parameter
template<int mfeat, int pchi> __global__
void logdensity_meas(double* meas, double* meas_unc, double* chi, double* logdens, pLogDensMeas LogDensityMeas,
		int ndata)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		double chi_i[pchi], meas_i[mfeat], meas_unc_i[mfeat];
		for (int j = 0; j < pchi; ++j) {
			chi_i[j] = chi[j * ndata + idata];
		}
		for (int j = 0; j < mfeat; ++j) {
			meas_i[j] = meas[j * ndata + idata];
			meas_unc_i[j] = meas_unc[j * ndata + idata];
		}
		logdens[idata] = LogDensityMeas(chi_i, meas_i, meas_unc_i, mfeat, pchi);
	}
}

// compute the conditional log-posterior density of the characteristics given the population parameter
template<int pchi, int dtheta> __global__
void logdensity_pop(double* chi, double* logdens, pLogDensPop LogDensityPop, int ndata)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		double chi_i[pchi];
		for (int j = 0; j < pchi; ++j) {
			chi_i[j] = chi[j * ndata + idata];
		}
		logdens[idata] = LogDensityPop(chi_i, c_theta, pchi, dtheta);
	}
}

#endif /* KERNELS_H__ */
