/*
 * LumFuncDaug.cuh
 *
 *  Created on: July 23, 2014
 *      Author: Janos M. Szalai-Gindl
 */

#ifndef LUMFUNCDAUG_CUH_
#define LUMFUNCDAUG_CUH_

// Kernel to compute the initial values of chi = flux.
template<int mfeat> __global__
void initial_flux_value(double* chi, double* meas, double* meas_unc, double* cholfact, double* logdens,
		int ndata, pLogDensMeas LogDensityMeas, curandState* devStates)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		for (int j = 0; j < pchi; ++j) {
			chi[idata + j * ndata] = 1.0; // initialize chi values to one
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

template <int mfeat,int pchi,int dtheta>
class LumFuncDaug: public DataAugmentation<mfeat, pchi, dtheta> {
public:

	LumFuncDaug(vecvec& meas, vecvec& meas_unc) : DataAugmentation<mfeat, pchi, dtheta>(meas, meas_unc) {}

	void Initialize() {
		// grab pointers to the device vector memory locations
		double* p_chi = thrust::raw_pointer_cast(&this->d_chi[0]);
		double* p_meas = thrust::raw_pointer_cast(&this->d_meas[0]);
		double* p_meas_unc = thrust::raw_pointer_cast(&this->d_meas_unc[0]);
		double* p_cholfact = thrust::raw_pointer_cast(&this->d_cholfact[0]);
		double* p_logdens = thrust::raw_pointer_cast(&this->d_logdens[0]);

		// set initial values for the characteristics. this will launch a CUDA kernel.
		initial_flux_value <mfeat> <<<this->nBlocks, this->nThreads>>>(p_chi, p_meas, p_meas_unc, p_cholfact, p_logdens,
				this->ndata, this->p_logdens_function, this->p_devStates);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		thrust::fill(this->d_naccept.begin(), this->d_naccept.end(), 0);
		this->current_iter = 1;
	}

	// launch the update kernel on the GPU
	virtual void Update() {
		// grab the pointers to the device memory locations
		double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
		double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
		double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
		double* p_cholfact = thrust::raw_pointer_cast(&d_cholfact[0]);
		double* p_logdens_meas = thrust::raw_pointer_cast(&d_logdens[0]);
		double* p_logdens_pop = p_Theta->GetDevLogDensPtr();
		int* p_naccept = thrust::raw_pointer_cast(&d_naccept[0]);

		double* p_distData = p_Theta->GetDistData();
		// grab host-side pointer function that compute the conditional posterior of characteristics|population
		pLogDensPopAux p_logdens_pop_aux_function = p_Theta->GetLogDensPopAuxPtr();

		// launch the kernel to update the characteristics on the GPU
		update_characteristic_aux <mfeat, pchi, dtheta> <<<nBlocks,nThreads>>>(p_meas, p_meas_unc, p_chi, p_cholfact,
				p_logdens_meas, p_logdens_pop, p_devStates, p_logdens_function, p_logdens_pop_aux_function, current_iter,
				p_naccept, ndata, p_distData);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	    current_iter++;
	}

	// setters and getters
	virtual void SetChi(dvector& chi, bool update_logdens = true) {
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
			double* p_distData = p_Theta->GetDistData();
			pLogDensPopAux p_LogDensPopAux = p_Theta->GetLogDensPopAuxPtr();
			logdensity_pop_aux <pchi, dtheta> <<<nBlocks,nThreads>>>(p_chi, p_logdens_pop, p_distData, p_LogDensPopAux, ndata);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
	}
};

#endif /* LUMFUNCDAUG_CUH_ */