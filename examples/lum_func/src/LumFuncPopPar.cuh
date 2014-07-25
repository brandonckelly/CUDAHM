/*
 * LumFuncPopPar.cuh
 *
 *  Created on: July 23, 2014
 *      Author: Janos M. Szalai-Gindl
 */

#ifndef LUMFUNCPOPPAR_CUH_
#define LUMFUNCPOPPAR_CUH_

#include "../CudaHM/parameters.cuh"

extern __constant__ pLogDensPopAux c_LogDensPopAux;

template <int mfeat, int pchi, int dtheta>
class LumFuncPopPar: public PopulationPar<mfeat, pchi, dtheta>
{
public:
	LumFuncPopPar(int ndata, thrust::device_vector<double>& d_distData) : PopulationPar<mfeat, pchi, dtheta>(), d_distData(d_distData)
	{
		// grab pointer to function that compute the log-density of characteristics|theta from device
		// __constant__ memory
	    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&p_logdensaux_function, c_LogDensPopAux, sizeof(c_LogDensPopAux)));
	}
	
	// calculate the initial value of the population parameters
	virtual void Initialize() {
		// first set initial values
		InitialValue();
		InitialCholFactor();
		// transfer initial value of theta to GPU constant memory
	    double* p_theta = thrust::raw_pointer_cast(&h_theta[0]);
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta*sizeof(*p_theta)));

		// get initial value of conditional log-posterior for theta|chi
		double* p_chi = Daug->GetDevChiPtr(); // grab pointer to Daug.d_chi
		double* p_logdens = thrust::raw_pointer_cast(&d_logdens[0]);
		double* p_distData = thrust::raw_pointer_cast(&d_distData[0]);
		logdensity_pop_aux <pchi, dtheta> <<<nBlocks,nThreads>>>(p_chi, p_logdens, p_distData, p_logdensaux_function, ndata);

		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	    current_logdens = thrust::reduce(d_logdens.begin(), d_logdens.end());

		// reset the number of MCMC iterations
		current_iter = 1;
		naccept = 0;
	}

	virtual void InitialValue() {
		// set initial value of theta to one
		h_theta[0] = -2.0;
		h_theta[1] = 1.0;
		h_theta[2] = 1.0;
		//thrust::fill(h_theta.begin(), h_theta.end(), 1.0);
	}

	// update the value of the population parameter value using a robust adaptive metropolis algorithm
	virtual void Update() {
		// get current conditional log-posterior of population
		double logdens_current = thrust::reduce(d_logdens.begin(), d_logdens.end());
		logdens_current += LogPrior(h_theta);

		// propose new value of population parameter
		hvector h_proposed_theta = Propose();

		// copy proposed theta to GPU constant memory
	    double* p_proposed_theta = thrust::raw_pointer_cast(&h_proposed_theta[0]);
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_proposed_theta, dtheta*sizeof(*p_proposed_theta)));

		// calculate log-posterior of new population parameter in parallel on the device
		const int ndata = Daug->GetDataDim();
		double* p_logdens_prop = thrust::raw_pointer_cast(&d_proposed_logdens[0]);
		double* p_distData = thrust::raw_pointer_cast(&d_distData[0]);
		
		logdensity_pop_aux <pchi, dtheta> <<<nBlocks,nThreads>>>(Daug->GetDevChiPtr(), p_logdens_prop, p_distData, p_logdensaux_function, ndata);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		double logdens_prop = thrust::reduce(d_proposed_logdens.begin(), d_proposed_logdens.end());

		logdens_prop += LogPrior(h_proposed_theta);

		// accept the proposed value?
		double metro_ratio = 0.0;
		bool accept = AcceptProp(logdens_prop, logdens_current, metro_ratio, 0.0, 0.0);
		if (accept) {
			h_theta = h_proposed_theta;
			thrust::copy(d_proposed_logdens.begin(), d_proposed_logdens.end(), d_logdens.begin());
			naccept++;
			current_logdens = logdens_prop;
		} else {
			// proposal rejected, so need to copy current theta back to constant memory
		    double* p_theta = thrust::raw_pointer_cast(&h_theta[0]);
			CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta*sizeof(*p_theta)));
			current_logdens = logdens_current;
		}

		// adapt the covariance matrix of the proposals
		AdaptProp(metro_ratio);
		current_iter++;
	}

	virtual void SetTheta(hvector& theta, bool update_logdens = true) {
		h_theta = theta;
	    double* p_theta = thrust::raw_pointer_cast(&theta[0]);
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta * sizeof(*p_theta)));
		if (update_logdens) {
			// update value of conditional log-posterior for theta|chi
			double* p_chi = Daug->GetDevChiPtr(); // grab pointer to Daug.d_chi
			double* p_logdens = thrust::raw_pointer_cast(&d_logdens[0]);
			double* p_distData = thrust::raw_pointer_cast(&d_distData[0]);
			logdensity_pop_aux <pchi, dtheta> <<<nBlocks,nThreads>>>(p_chi, p_logdens, p_distData, p_logdensaux_function, ndata);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			current_logdens = thrust::reduce(d_logdens.begin(), d_logdens.end());
            current_logdens += LogPrior(h_theta);
		}
	}

	virtual double* GetDistData() { return thrust::raw_pointer_cast(&d_distData[0]); }
	virtual pLogDensPopAux GetLogDensPopAuxPtr() { return p_logdensaux_function; }

protected:
	thrust::device_vector<double> d_distData;
	// pointer to device-side function that compute the conditional log-posterior of characteristics|population
	pLogDensPopAux p_logdensaux_function;

};

#endif /* LUMFUNCPOPPAR_CUH_ */