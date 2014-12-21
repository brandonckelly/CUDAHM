/*
 * LumFuncDaug.cuh
 *
 *  Created on: July 23, 2014
 *      Author: Janos M. Szalai-Gindl
 */

#ifndef LUMFUNCDAUG_CUH_
#define LUMFUNCDAUG_CUH_

#include "LumFuncDist.cuh"

extern __constant__ pLogDensPopAux c_LogDensPopAux;

// Kernel to compute the initial values of chi = flux.
template<int mfeat, int pchi> __global__
void initial_flux_value(double* chi, double* meas, double* meas_unc, double* cholfact, double* logdens,
		int ndata, pLogDensMeas LogDensityMeas, curandState* devStates)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		for (int j = 0; j < pchi; ++j) {
			chi[idata + j * ndata] = meas[idata + j * ndata]; // initialize chi values to measurements
		}

		// set initial covariance matrix of the chi proposals as the identity matrix
		int diag_index = 0;
		for (int j=0; j<pchi; j++) {
			cholfact[idata + ndata * diag_index] = 1.0;
			diag_index += j + 2;  // cholesky factor is lower diagonal
		}
		// copy value to registers
		double this_chi[pchi];
		for (int j = 0; j < pchi; ++j) {
			this_chi[j] = chi[j * ndata + idata];
		}
		double local_meas[mfeat], local_meas_unc[mfeat];
		for (int j = 0; j < mfeat; ++j) {
			local_meas[j] = meas[j * ndata + idata];
			local_meas_unc[j] = meas_unc[j * ndata + idata];
		}
		logdens[idata] = LogDensityMeas(this_chi, local_meas, local_meas_unc);
	}
}

// kernel to update the values of the characteristics in parallel on the GPU
// this variation uses auxiliary data for determination of log p(chi_i | theta)
template<int mfeat, int pchi, int dtheta> __global__
void update_characteristic_aux(double* meas, double* meas_unc, double* chi, double* cholfact,
double* logdens_meas, double* logdens_pop, curandState* devStates, pLogDensMeas LogDensityMeas,
pLogDensPopAux LogDensityPop, int current_iter, int* naccept, int ndata, double* auxdata)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		curandState localState = devStates[idata]; // grab state of this random number generator

		// copy values for this data point to registers for speed
		double snorm_deviate[pchi], scaled_proposal[pchi], proposed_chi[pchi], local_chi[pchi];
		const int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
		double local_cholfact[dim_cholfact];
		int cholfact_index = 0;
		for (int j = 0; j < pchi; ++j) {
			local_chi[j] = chi[j * ndata + idata];
			for (int k = 0; k < (j + 1); ++k) {
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
		double logdens_meas_prop = LogDensityMeas(proposed_chi, local_meas, local_meas_unc);
		double auxdata_i = auxdata[idata];
		double logdens_pop_prop = LogDensityPop(proposed_chi, c_theta, auxdata_i);
		double logpost_prop = logdens_meas_prop + logdens_pop_prop;

		/*
		* Uncomment this bit of code for help when debugging.
		*
		if (idata == 0) {
		printf("current iter, idata, mfeat, pchi, dtheta: %i, %i, %i, %i, %i\n", current_iter, idata, mfeat, pchi, dtheta);
		printf("  measurements: %g, %g, %g, %g, %g\n", local_meas[0], local_meas[1], local_meas[2], local_meas[3], local_meas[4]);
		printf("  measurement sigmas: %g, %g, %g, %g, %g\n", local_meas_unc[0], local_meas_unc[1], local_meas_unc[2],
		local_meas_unc[3], local_meas_unc[4]);
		printf("  cholfact: %g, %g, %g, %g, %g, %g\n", local_cholfact[0], local_cholfact[1], local_cholfact[2], local_cholfact[3],
		local_cholfact[4], local_cholfact[5]);
		printf("  current chi: %g, %g, %g\n", local_chi[0], local_chi[1], local_chi[2]);
		printf("  proposed chi: %g, %g, %g\n", proposed_chi[0], proposed_chi[1], proposed_chi[2]);
		printf("  current logdens_meas, logdens_pop: %g, %g\n", logdens_meas[idata], logdens_pop[idata]);
		printf("  proposed logdens_meas, logdens_pop: %g, %g\n", logdens_meas_prop, logdens_pop_prop);
		printf("\n");
		}
		*/

		// accept the proposed value of the characteristic?
		double logdens_meas_i = logdens_meas[idata];
		double logdens_pop_i = logdens_pop[idata];
		double logpost_current = logdens_meas_i + logdens_pop_i;
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
		// printf("current iter, Accept, idata: %d, %d, %d\n", current_iter, accept, idata);
		for (int j = 0; j<pchi; j++) {
			chi[ndata * j + idata] = accept ? proposed_chi[j] : local_chi[j];
		}
		logdens_meas[idata] = accept ? logdens_meas_prop : logdens_meas_i;
		logdens_pop[idata] = accept ? logdens_pop_prop : logdens_pop_i;
		naccept[idata] += accept;

	}

}

template <int mfeat,int pchi,int dtheta>
class LumFuncDaug: public DataAugmentation<mfeat, pchi, dtheta> {
public:

	LumFuncDaug(vecvec& meas, vecvec& meas_unc, LumFuncDist& lumFuncDist) : DataAugmentation<mfeat, pchi, dtheta>(meas, meas_unc), lumFuncDist(lumFuncDist)
	{
		// grab pointer to function that compute the log-density of characteristics|theta from device
		// __constant__ memory
		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&p_logdensaux_function, c_LogDensPopAux, sizeof(c_LogDensPopAux)));
	}

	void Initialize() {
		// grab pointers to the device vector memory locations
		double* p_chi = thrust::raw_pointer_cast(&this->d_chi[0]);
		double* p_meas = thrust::raw_pointer_cast(&this->d_meas[0]);
		double* p_meas_unc = thrust::raw_pointer_cast(&this->d_meas_unc[0]);
		double* p_cholfact = thrust::raw_pointer_cast(&this->d_cholfact[0]);
		double* p_logdens = thrust::raw_pointer_cast(&this->d_logdens[0]);

		// set initial values for the characteristics. this will launch a CUDA kernel.
		initial_flux_value <mfeat,pchi> <<<this->nBlocks, this->nThreads>>>(p_chi, p_meas, p_meas_unc, p_cholfact, p_logdens,
				this->ndata, this->p_logdens_function, this->p_devStates);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		thrust::fill(this->d_naccept.begin(), this->d_naccept.end(), 0);
		this->current_iter = 1;
	}

	// launch the update kernel on the GPU
	void Update() {
		// grab the pointers to the device memory locations
		double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
		double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
		double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
		double* p_cholfact = thrust::raw_pointer_cast(&d_cholfact[0]);
		double* p_logdens_meas = thrust::raw_pointer_cast(&d_logdens[0]);
		double* p_logdens_pop = p_Theta->GetDevLogDensPtr();
		int* p_naccept = thrust::raw_pointer_cast(&d_naccept[0]);

		double* p_distData = lumFuncDist.GetDistData();

		// launch the kernel to update the characteristics on the GPU
		update_characteristic_aux <mfeat, pchi, dtheta> <<<nBlocks,nThreads>>>(p_meas, p_meas_unc, p_chi, p_cholfact,
			p_logdens_meas, p_logdens_pop, p_devStates, p_logdens_function, p_logdensaux_function, current_iter,
				p_naccept, ndata, p_distData);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		current_iter++;
	}

	// setters and getters
	// NOT USED
	void SetChi(dvector& chi, bool update_logdens = true) {
		d_chi = chi;
		if (update_logdens) {
			// update the posteriors for the new values of the characteristics
			double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
			double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
			double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
			double* p_logdens_meas = thrust::raw_pointer_cast(&d_logdens[0]);
			// first update the posteriors of measurements | characteristics
			logdensity_meas <mfeat, pchi> <<<nBlocks,nThreads>>>(p_meas, p_meas_unc, p_chi, p_logdens_meas, p_logdens_function,
					ndata);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			double* p_logdens_pop = p_Theta->GetDevLogDensPtr();
			// no update the posteriors of the characteristics | population parameter
			double* p_distData = lumFuncDist.GetDistData();
			logdensity_pop_aux <pchi, dtheta> << <nBlocks, nThreads >> >(p_chi, p_logdens_pop, p_distData, p_logdensaux_function, ndata);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
	}

protected:
	LumFuncDist& lumFuncDist;
	// pointer to device-side function that compute the conditional log-posterior of characteristics|population
	pLogDensPopAux p_logdensaux_function;
};

#endif /* LUMFUNCDAUG_CUH_ */