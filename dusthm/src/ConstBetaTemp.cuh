/*
 * ConstBetaTemp.cuh
 *
 *  Created on: Mar 26, 2014
 *      Author: brandonkelly
 */

#ifndef CONSTBETATEMP_CUH_
#define CONSTBETATEMP_CUH_

#include "parameters.cuh"

// Kernel to compute the initial values of chi = (log C, beta, log T). Does this by randomly generating values within a range
// of reasonable physical values for far-IR SEDs of dust in galactic starless cores.
template<int mfeat> __global__
void initial_cbt_value(double* chi, double* meas, double* meas_unc, double* cholfact, double* logdens,
		int ndata, pLogDensMeas LogDensityMeas, curandState* devStates)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		curandState localState = devStates[idata]; // grab state of this random number generator

		double cbt_mu[3] = {15.0, 2.0, log(15.0)};
		double cbt_sigma[3] = {4.6, 0.5, 0.33};
		for (int j = 0; j < 3; ++j) {
			// randomly initialize chi = (log C, beta, log T) from a normal distribution
			chi[idata + j * ndata] = cbt_mu[j] + curand_normal_double(&localState);
		}

		// set initial covariance matrix of the chi proposals as the identity matrix
		int diag_index = 0;
		for (int j=0; j<3; j++) {
			cholfact[idata + ndata * diag_index] = 1.0;
			diag_index += j + 2;  // cholesky factor is lower diagonal
		}
		// copy value to registers before computing initial log-density
		double this_chi[3];
		for (int j = 0; j < pchi; ++j) {
			this_chi[j] = chi[j * ndata + idata];
		}
		double local_meas[mfeat], local_meas_unc[mfeat];
		for (int j = 0; j < mfeat; ++j) {
			local_meas[j] = meas[j * ndata + idata];
			local_meas_unc[j] = meas_unc[j * ndata + idata];
		}
		logdens[idata] = LogDensityMeas(this_chi, local_meas, local_meas_unc);

		// copy local RNG state back to global memory
		devStates[idata] = localState;
	}
}

// Subclass the DataAugmentation class to override the method to generate the initial values of chi = (log C, beta, log T)
template <mfeat, pchi, dtheta>
class ConstBetaTemp: public DataAugmentation<mfeat, pchi, dtheta> {
public:
	void Initialize() {
		// grab pointers to the device vector memory locations
		double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
		double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
		double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
		double* p_cholfact = thrust::raw_pointer_cast(&d_cholfact[0]);
		double* p_logdens = thrust::raw_pointer_cast(&d_logdens[0]);

		// set initial values for the characteristics. this will launch a CUDA kernel.
		initial_cbt_value <mfeat> <<<nBlocks,nThreads>>>(p_chi, p_meas, p_meas_unc, p_cholfact, p_logdens,
				ndata, p_logdens_function, p_devStates);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		thrust::fill(d_naccept.begin(), d_naccept.end(), 0);
		current_iter = 1;
	}
};

#endif /* CONSTBETATEMP_CUH_ */

