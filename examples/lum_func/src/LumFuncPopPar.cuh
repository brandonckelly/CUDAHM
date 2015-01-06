/*
* LumFuncPopPar.cuh
*
*  Created on: July 23, 2014
*      Author: Janos M. Szalai-Gindl
*/

#ifndef LUMFUNCPOPPAR_CUH_
#define LUMFUNCPOPPAR_CUH_

#include "../../../mwg/src/parameters.cuh"
#include "LumFuncDist.cuh"

typedef double(*pLogDensPopAux)(double*, double*, double);

extern __constant__ pLogDensPopAux c_LogDensPopAux;

boost::random::normal_distribution<> snorm_sigma_1(0.0, 1.0e12); // normal distribution with st dev 1.0e12 for proposal

// compute the conditional log-posterior density of the characteristics given the population parameter
// and auxiliary data which are needed to density function determination
template<int pchi, int dtheta> __global__
void logdensity_pop_aux(double* chi, double* logdens, double* auxdata, pLogDensPopAux LogDensityPop, int ndata)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		double chi_i[pchi];
		for (int j = 0; j < pchi; ++j) {
			chi_i[j] = chi[j * ndata + idata];
		}
		double auxdata_i = auxdata[idata];
		logdens[idata] = LogDensityPop(chi_i, c_theta, auxdata_i);
	}
}

template <int mfeat, int pchi, int dtheta>
class LumFuncPopPar : public PopulationPar<mfeat, pchi, dtheta>
{
public:
	LumFuncPopPar(int ndata, LumFuncDist& lumFuncDist) : PopulationPar<mfeat, pchi, dtheta>(), lumFuncDist(lumFuncDist)
	{
		// grab pointer to function that compute the log-density of characteristics|theta from device
		// __constant__ memory
		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&p_logdensaux_function, c_LogDensPopAux, sizeof(c_LogDensPopAux)));
	}

	// calculate the initial value of the population parameters
	void Initialize() {
		// first set initial values
		InitialValue();
		InitialCholFactor();
		// transfer initial value of theta to GPU constant memory
		double* p_theta = thrust::raw_pointer_cast(&h_theta[0]);
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta*sizeof(*p_theta)));

		// get initial value of conditional log-posterior for theta|chi
		double* p_chi = Daug->GetDevChiPtr(); // grab pointer to Daug.d_chi
		double* p_logdens = thrust::raw_pointer_cast(&d_logdens[0]);
		double* p_distData = lumFuncDist.GetDistData();
		logdensity_pop_aux <pchi, dtheta> << <nBlocks, nThreads >> >(p_chi, p_logdens, p_distData, p_logdensaux_function, ndata);

		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		current_logdens = thrust::reduce(d_logdens.begin(), d_logdens.end());

		// reset the number of MCMC iterations
		current_iter = 1;
		naccept = 0;
	}

	void InitialValue() {
		// set initial value of theta
		h_theta[0] = -1.41;
		h_theta[1] = 4.0;
		h_theta[2] = 5.8;
	}

	void InitialCholFactor() {
		// set initial covariance matrix of the theta proposals as the identity matrix
		thrust::fill(cholfact.begin(), cholfact.end(), 0.0);
		cholfact[0] = 1.0;
		cholfact[2] = 1.0;
		cholfact[5] = 1.0;
	}

	// update the value of the population parameter value using a robust adaptive metropolis algorithm
	void Update() {
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
		double* p_distData = lumFuncDist.GetDistData();

		logdensity_pop_aux <pchi, dtheta> << <nBlocks, nThreads >> >(Daug->GetDevChiPtr(), p_logdens_prop, p_distData, p_logdensaux_function, ndata);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		double logdens_prop = thrust::reduce(d_proposed_logdens.begin(), d_proposed_logdens.end());

		logdens_prop += LogPrior(h_proposed_theta);

		//double* p_chi = Daug->GetChi();

		//double erf_sum = 0.0;
		//for (int idx = 0; idx < ndata; ++idx)
		//{
		//	double chi_i = p_chi[idx];
		//	double sigma = 1e-10;
		//	double fluxLimit = 1e-13;
		//	erf_sum += 0.5 * (1.0 + erf((chi_i - fluxLimit) / (sigma * sqrt(2))));
		//}
		//delete[] p_chi;

		//double p_logC_given_theta = (ndata - 1) * log((1.0 / (double) ndata) * erf_sum);

		//// the effect of flux limit:
		//logdens_prop -= p_logC_given_theta;

		// accept the proposed value?
		double metro_ratio = 0.0;
		bool accept = AcceptProp(logdens_prop, logdens_current, metro_ratio, 0.0, 0.0);
		if (accept) {
			h_theta = h_proposed_theta;
			thrust::copy(d_proposed_logdens.begin(), d_proposed_logdens.end(), d_logdens.begin());
			naccept++;
			current_logdens = logdens_prop;
		}
		else {
			// proposal rejected, so need to copy current theta back to constant memory
			double* p_theta = thrust::raw_pointer_cast(&h_theta[0]);
			CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta*sizeof(*p_theta)));
			current_logdens = logdens_current;
		}

		// adapt the covariance matrix of the proposals
		AdaptProp(metro_ratio);
		current_iter++;
	}
	
	// NOT USED
	void SetTheta(hvector& theta, bool update_logdens = true) {
		h_theta = theta;
		double* p_theta = thrust::raw_pointer_cast(&theta[0]);
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta * sizeof(*p_theta)));
		if (update_logdens) {
			// update value of conditional log-posterior for theta|chi
			double* p_chi = Daug->GetDevChiPtr(); // grab pointer to Daug.d_chi
			double* p_logdens = thrust::raw_pointer_cast(&d_logdens[0]);
			double* p_distData = lumFuncDist.GetDistData();
			logdensity_pop_aux <pchi, dtheta> << <nBlocks, nThreads >> >(p_chi, p_logdens, p_distData, p_logdensaux_function, ndata);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			current_logdens = thrust::reduce(d_logdens.begin(), d_logdens.end());
			current_logdens += LogPrior(h_theta);
		}
	}

	double min_double()
	{
		// const unsigned long long ieee754mindouble = 0xffefffffffffffff;
		// return __longlong_as_double(ieee754mindouble);
		// we choose the next double for minimal double because of technical reason:
		// (If we summarize (more than one) ieee754mindoubles we get NaN result.)
		return -1.797693e+250;
	}

	double LogPrior(hvector theta) {
		double negative_infinity = -std::numeric_limits<double>::infinity();
		double gammaTheta = theta[0];
		double lScaleTheta = theta[1];
		double uScaleTheta = theta[2];
		double result;
		if ((gammaTheta < 0) && (gammaTheta > -2) && (lScaleTheta < uScaleTheta))
		{
			result = log(0.90322) + log(lScaleTheta) - 2 * log(uScaleTheta) - log(1 + gammaTheta * gammaTheta);
		}
		else
		{
			result = min_double();
		}
		return result;
	}

	hvector Propose() {
		// get the unit proposal
		for (int k = 0; k<dtheta; k++) {
			snorm_deviate[k] = snorm(rng);
		}

		if ((current_iter == 1) || (current_iter % 1000 == 0))
		{
			int cholfact_index = 0;
			std::string line = "";
			printf("%d cholfact:\n", current_iter);
			for (int j = 0; j < dtheta; j++) {
				for (int k = 0; k < (j + 1); k++) {
					line += std::to_string(cholfact[cholfact_index]) + " ";
					cholfact_index++;
				}
				line += "\n";
				printf(line.c_str());
				line = "";
			}
		}

		// transform unit proposal so that is has a multivariate normal distribution
		hvector proposed_theta(dtheta);
		thrust::fill(scaled_proposal.begin(), scaled_proposal.end(), 0.0);
		int cholfact_index = 0;
		for (int j = 0; j<dtheta; j++) {
			for (int k = 0; k<(j + 1); k++) {
				// cholfact is lower-diagonal matrix stored as a 1-d array
				scaled_proposal[j] += cholfact[cholfact_index] * snorm_deviate[k];
				cholfact_index++;
			}
			proposed_theta[j] = h_theta[j] + scaled_proposal[j];
		}

		return proposed_theta;
	}

protected:
	LumFuncDist& lumFuncDist;
	// pointer to device-side function that compute the conditional log-posterior of characteristics|population
	pLogDensPopAux p_logdensaux_function;

};

#endif /* LUMFUNCPOPPAR_CUH_ */