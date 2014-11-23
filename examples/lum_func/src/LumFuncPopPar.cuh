/*
* LumFuncPopPar.cuh
*
*  Created on: July 23, 2014
*      Author: Janos M. Szalai-Gindl
*/

#ifndef LUMFUNCPOPPAR_CUH_
#define LUMFUNCPOPPAR_CUH_

#include "../../../mwg/src/parameters.cuh"

extern __constant__ pLogDensPopAux c_LogDensPopAux;

template <int mfeat, int pchi, int dtheta>
class LumFuncPopPar : public PopulationPar<mfeat, pchi, dtheta>
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
		logdensity_pop_aux <pchi, dtheta> << <nBlocks, nThreads >> >(p_chi, p_logdens, p_distData, p_logdensaux_function, ndata);

		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		current_logdens = thrust::reduce(d_logdens.begin(), d_logdens.end());

		// reset the number of MCMC iterations
		current_iter = 1;
		naccept = 0;
	}

	virtual void InitialValue() {
		// set initial value of theta
		h_theta[0] = -1.3;
		h_theta[1] = 5.0;
		h_theta[2] = 110.0;
	}

	// update the value of the population parameter value using a robust adaptive metropolis algorithm
	virtual void Update() {
		//TestEstimation();
		// get current conditional log-posterior of population
		double logdens_current = thrust::reduce(d_logdens.begin(), d_logdens.end());
		logdens_current += LogPrior(h_theta);

		/*if (_isnan(logdens_current) || !_finite(logdens_current))
		{
		logdens_current = logdens_current;
		}*/

		// propose new value of population parameter
		hvector h_proposed_theta = Propose();

		// copy proposed theta to GPU constant memory
		double* p_proposed_theta = thrust::raw_pointer_cast(&h_proposed_theta[0]);
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_proposed_theta, dtheta*sizeof(*p_proposed_theta)));

		// calculate log-posterior of new population parameter in parallel on the device
		const int ndata = Daug->GetDataDim();
		double* p_logdens_prop = thrust::raw_pointer_cast(&d_proposed_logdens[0]);
		double* p_distData = thrust::raw_pointer_cast(&d_distData[0]);

		logdensity_pop_aux <pchi, dtheta> << <nBlocks, nThreads >> >(Daug->GetDevChiPtr(), p_logdens_prop, p_distData, p_logdensaux_function, ndata);
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

	void TestEstimation() {
		const int ndata = Daug->GetDataDim();
		double* p_logdens_prop = thrust::raw_pointer_cast(&d_proposed_logdens[0]);
		double* p_distData = thrust::raw_pointer_cast(&d_distData[0]);

		std::vector<double> flux_true(ndata);
		std::string fluxfile = "C:/temp/data/beta-1.5_l1.0_u100.0_sig1e-10_n1000/fluxes_cnt_1000.dat";

		std::ifstream input_file(fluxfile.c_str());
		flux_true.resize(ndata);
		for (int i = 0; i < ndata; ++i) {
			input_file >> flux_true[i];
		}
		input_file.close();

		// copy input data to data members
		hvector h_flux(ndata);
		dvector d_flux;
		for (int i = 0; i < ndata; ++i) {
			h_flux[i] = flux_true[i];
		}

		// copy data from host to device
		d_flux = h_flux;

		Daug->SetChi(d_flux, true);

		hvector h_theta(dtheta);
		h_theta[0] = -1.499;
		h_theta[1] = 1.001;
		h_theta[2] = 100.001;

		SetTheta(h_theta, true);

		std::cout << "current_logdens " << current_logdens << std::endl;

		Daug->SetChi(d_flux, true);

		h_theta[0] = -1.99937;
		h_theta[1] = 2.5301e-015;
		h_theta[2] = 99.8111;

		SetTheta(h_theta, true);

		std::cout << "current_logdens " << current_logdens << std::endl;
	}

	// NOT USED
	virtual void SetTheta(hvector& theta, bool update_logdens = true) {
		h_theta = theta;
		double* p_theta = thrust::raw_pointer_cast(&theta[0]);
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta * sizeof(*p_theta)));
		if (update_logdens) {
			// update value of conditional log-posterior for theta|chi
			double* p_chi = Daug->GetDevChiPtr(); // grab pointer to Daug.d_chi
			double* p_logdens = thrust::raw_pointer_cast(&d_logdens[0]);
			double* p_distData = thrust::raw_pointer_cast(&d_distData[0]);
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

	virtual double LogPrior(hvector theta) {
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

	virtual hvector Propose() {
		// get the unit proposal
		for (int k = 0; k<dtheta; k++) {
			snorm_deviate[k] = snorm(rng);
		}

		//snorm_deviate[0] = snorm(rng);

		//snorm_deviate[1] = snorm_sigma_1(rng);

		//snorm_deviate[2] = snorm_sigma_1(rng);

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


	virtual double* GetDistData() { return thrust::raw_pointer_cast(&d_distData[0]); }
	virtual pLogDensPopAux GetLogDensPopAuxPtr() { return p_logdensaux_function; }

protected:
	thrust::device_vector<double> d_distData;
	// pointer to device-side function that compute the conditional log-posterior of characteristics|population
	pLogDensPopAux p_logdensaux_function;

};

#endif /* LUMFUNCPOPPAR_CUH_ */