/*
 * parameters.cuh
 *
 *  Created on: Jul 28, 2013
 *      Author: brandonkelly
 */

#ifndef _DATA_AUGMENTATION_HPP__
#define _DATA_AUGMENTATION_HPP__

// macro for checking for CUDA errors
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

// Cuda Includes
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Standard includes
#include <cmath>
#include <vector>
#include <stdio.h>

// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Boost includes
#include <boost/shared_ptr.hpp>

// Local includes
#include "kernels.cuh"

/*
 * POINTERS TO FUNCTIONS THAT ARE NEEDED TO COMPUTE THE CONDITIONAL LOG-POSTERIORS. THESE MUST BE DEFINED ELSEWHERE TO
 * POINT TO THE USER-SUPPLIED FUNCTIONS FOR COMPUTING THE LOG-POSTERIORS.
 */
extern __constant__ pLogDensMeas c_LogDensMeas;
extern __constant__ pLogDensPop c_LogDensPop;

// Global constants for MCMC sampler
const double target_rate = 0.4; // MCMC sampler target acceptance rate
const double decay_rate = 0.667; // decay rate of robust adaptive metropolis algorithm

// convenience typedefs
typedef std::vector<std::vector<double> > vecvec;
typedef thrust::host_vector<double> hvector;
typedef thrust::device_vector<double> dvector;

template <int mfeat, int pchi, int dtheta> class PopulationPar; // forward declaration


// class for a data augmentation.
template <int mfeat, int pchi, int dtheta>
class DataAugmentation
{
public:
	// Constructor
	DataAugmentation(vecvec& meas, vecvec& meas_unc, dim3& nB, dim3& nT) : ndata(meas.size()), nBlocks(nB), nThreads(nT)
		{
			_SetArraySizes();

			// copy input data to data members
			for (int j = 0; j < mfeat; ++j) {
				for (int i = 0; i < ndata; ++i) {
					h_meas[ndata * j + i] = meas[i][j];
					h_meas_unc[ndata * j + i] = meas_unc[i][j];
				}
			}
			// copy data from host to device
			d_meas = h_meas;
			d_meas_unc = h_meas_unc;

			thrust::fill(d_cholfact.begin(), d_cholfact.end(), 0.0);

			// Allocate memory on GPU for RNG states
			CUDA_CHECK_RETURN(cudaMalloc((void **)&p_devStates, nThreads.x * nBlocks.x * sizeof(curandState)));
			// Initialize the random number generator states on the GPU
			initialize_rng<<<nBlocks,nThreads>>>(p_devStates);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			// Wait until RNG stuff is done running on the GPU, make sure everything went OK
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());

			// grab pointer to function that compute the log-density of measurements|characteristics from device
			// __constant__ memory
		    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&p_logdens_function, c_LogDensMeas, sizeof(c_LogDensMeas)));

			save_trace = true;
		}

	virtual ~DataAugmentation() { cudaFree(p_devStates); }

	void Initialize() {
		// grab pointers to the device vector memory locations
		double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
		double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
		double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
		double* p_cholfact = thrust::raw_pointer_cast(&d_cholfact[0]);
		double* p_logdens = thrust::raw_pointer_cast(&d_logdens[0]);

		// set initial values for the characteristics. this will launch a CUDA kernel.
		initial_chi_value <mfeat, pchi> <<<nBlocks,nThreads>>>(p_chi, p_meas, p_meas_unc, p_cholfact, p_logdens,
				ndata, p_logdens_function);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		thrust::fill(d_naccept.begin(), d_naccept.end(), 0);
		current_iter = 1;
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

		// grab host-side pointer function that compute the conditional posterior of characteristics|population
		pLogDensPop p_logdens_pop_function = p_Theta->GetLogDensPopPtr();

		// launch the kernel to update the characteristics on the GPU
		update_characteristic <mfeat, pchi, dtheta> <<<nBlocks,nThreads>>>(p_meas, p_meas_unc, p_chi, p_cholfact,
				p_logdens_meas, p_logdens_pop, p_devStates, p_logdens_function, p_logdens_pop_function, current_iter,
				p_naccept, ndata);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	    current_iter++;
	}

	void ResetAcceptance() {
		thrust::fill(d_naccept.begin(), d_naccept.end(), 0);
	}

	// setters and getters
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
			pLogDensPop p_LogDensPop = p_Theta->GetLogDensPopPtr();
			logdensity_pop <pchi, dtheta> <<<nBlocks,nThreads>>>(p_chi, p_logdens_pop, p_LogDensPop, ndata);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
	}

	// make sure that the data augmentation knows about the population parameters
	void SetPopulationPtr(boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > t) { p_Theta = t; }

	void SetLogDens(dvector& logdens) {
		d_logdens = logdens;
	}
	void SetCholFact(dvector& cholfact) {
		d_cholfact = cholfact;
	}

	void SetSaveTrace(bool dosave) { save_trace = dosave; }

	bool SaveTrace() { return save_trace; }

	// return the value of the characteristic in a std::vector of std::vectors for convenience
	vecvec GetChi() {
		hvector h_chi = d_chi;  // first grab the values from the GPU
		vecvec chi(ndata);
		for (int i = 0; i < ndata; ++i) {
			// organize values into a 2-d array of dimensions ndata x pchi
			std::vector<double> chi_i(pchi);
			for (int j = 0; j < pchi; ++j) {
				chi_i[j] = h_chi[ndata * j + i];
			}
			chi[i] = chi_i;
		}
		return chi;
	}

	hvector GetHostLogDens() {
		hvector h_logdens = d_logdens;
		return h_logdens;
	}

	double GetLogDens() {
		double logdensity = thrust::reduce(d_logdens.begin(), d_logdens.end());
		return logdensity;
	}
	dvector GetDevLogDens() { return d_logdens; } // return the summed log-densities for y|chi
	double* GetDevLogDensPtr() { return thrust::raw_pointer_cast(&d_logdens[0]); }
	hvector GetHostChi() {
		hvector h_chi = d_chi;
		return h_chi;
	}
	dvector GetDevChi() { return d_chi; }
	double* GetDevChiPtr() { return thrust::raw_pointer_cast(&d_chi[0]); }
	int GetDataDim() { return ndata; }
	int GetChiDim() { return pchi; }
	boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > GetPopulationPtr() { return p_Theta; }
	pLogDensMeas GetLogDensMeasPtr() { return p_logdens_function; }
	thrust::host_vector<int> GetNaccept() {
		hvector h_naccept = d_naccept;
		return h_naccept;
	}

protected:
	// set the sizes of the data members
	void _SetArraySizes() {
		h_meas.resize(ndata * mfeat);
		d_meas.resize(ndata * mfeat);
		h_meas_unc.resize(ndata * mfeat);
		d_meas_unc.resize(ndata * mfeat);
		d_logdens.resize(ndata);
		d_chi.resize(ndata * pchi);
		int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
		d_cholfact.resize(ndata * dim_cholfact);
		d_naccept.resize(ndata);
	}
	// measurements and their uncertainties
	hvector h_meas;
	hvector h_meas_unc;
	dvector d_meas;
	dvector d_meas_unc;
	int ndata;
	// characteristics
	dvector d_chi;
	// population-level parameters
	boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > p_Theta;
	// logarithm of conditional posterior densities, y | chi
	dvector d_logdens;
	// cholesky factors of Metropolis proposal covariance matrix
	dvector d_cholfact;
	// state of parallel random number generator on the device
	curandState* p_devStates;
	// CUDA kernel launch specifications
	dim3& nBlocks;
	dim3& nThreads;
	// pointer to device-side function that compute the conditional log-posterior of measurements|characteristics
	pLogDensMeas p_logdens_function;
	// MCMC sampler parameters
	int current_iter;
	thrust::device_vector<int> d_naccept;
	bool save_trace;
};

// class for a population level parameter
template <int mfeat, int pchi, int dtheta>
class PopulationPar
{
public:
	// constructors
	PopulationPar(dim3& nB, dim3& nT) : nBlocks(nB), nThreads(nT)
	{
		h_theta.resize(dtheta);
		snorm_deviate.resize(dtheta);
		scaled_proposal.resize(dtheta);
		const int dim_cholfact = dtheta * dtheta - ((dtheta - 1) * dtheta) / 2;
		cholfact.resize(dim_cholfact);
		current_logdens = -1e300;
		current_iter = 0;
		naccept = 0;
		// grab pointer to function that compute the log-density of characteristics|theta from device
		// __constant__ memory
	    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&p_logdens_function, c_LogDensPop, sizeof(c_LogDensPop)));
	}

	virtual void InitialValue() {
		// set initial value of theta to zero
		thrust::fill(h_theta.begin(), h_theta.end(), 0.0);
	}

	// calculate the initial value of the population parameters
	virtual void Initialize() {
		// first set initial values
		InitialValue();

		// transfer initial value of theta to GPU constant memory
	    double* p_theta = thrust::raw_pointer_cast(&h_theta[0]);
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta*sizeof(*p_theta)));

		// set initial covariance matrix of the theta proposals as the identity matrix
		thrust::fill(cholfact.begin(), cholfact.end(), 0.0);
		int diag_index = 0;
		for (int k=0; k<dtheta; k++) {
			cholfact[diag_index] = 1.0;
			diag_index += k + 2;
		}

		// get initial value of conditional log-posterior for theta|chi
		double* p_chi = Daug->GetDevChiPtr(); // grab pointer to Daug.d_chi
		double* p_logdens = thrust::raw_pointer_cast(&d_logdens[0]);
		logdensity_pop <pchi, dtheta> <<<nBlocks,nThreads>>>(p_chi, p_logdens, p_logdens_function, ndata);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	    current_logdens = thrust::reduce(d_proposed_logdens.begin(), d_proposed_logdens.end());

		// reset the number of MCMC iterations
		current_iter = 1;
		naccept = 0;
	}

	// return the log-prior of the population parameters
	virtual double LogPrior(hvector theta) { return 0.0; }

	// propose a new value of the population parameters
	virtual hvector Propose() {
	    // get the unit proposal
	    for (int k=0; k<dtheta; k++) {
	        snorm_deviate[k] = snorm(rng);
	    }

	    // transform unit proposal so that is has a multivariate normal distribution
	    hvector proposed_theta(dtheta);
	    thrust::fill(scaled_proposal.begin(), scaled_proposal.end(), 0.0);
	    int cholfact_index = 0;
	    for (int j=0; j<dtheta; j++) {
	        for (int k=0; k<(j+1); k++) {
	        	// cholfact is lower-diagonal matrix stored as a 1-d array
	            scaled_proposal[j] += cholfact[cholfact_index] * snorm_deviate[k];
	            cholfact_index++;
	        }
	        proposed_theta[j] = h_theta[j] + scaled_proposal[j];
	    }

	    return proposed_theta;
	}

	// adapt the covariance matrix (i.e., the cholesky factors) of the theta proposals
	virtual void AdaptProp(double metro_ratio) {
		double unit_norm = 0.0;
	    for (int j=0; j<dtheta; j++) {
	    	unit_norm += snorm_deviate[j] * snorm_deviate[j];
	    }
	    unit_norm = sqrt(unit_norm);
	    double decay_sequence = 1.0 / std::pow(current_iter, decay_rate);
	    double scaled_coef = sqrt(decay_sequence * fabs(metro_ratio - target_rate)) / unit_norm;
	    for (int j=0; j<dtheta; j++) {
	        scaled_proposal[j] *= scaled_coef;
	    }

	    bool downdate = (metro_ratio < target_rate);
	    double* p_cholfact = thrust::raw_pointer_cast(&cholfact[0]);
	    double* p_scaled_proposal = thrust::raw_pointer_cast(&scaled_proposal[0]);
	    // rank-1 update of the cholesky factor
	    chol_update_r1(p_cholfact, p_scaled_proposal, dtheta, downdate);
	}

	// calculate whether to accept or reject the metropolist-hastings proposal
	bool AcceptProp(double logdens_prop, double logdens_current, double& ratio, double forward_dens = 0.0,
			double backward_dens = 0.0) {
	    double lograt = logdens_prop - forward_dens - (logdens_current - backward_dens);
	    lograt = std::min(lograt, 0.0);
	    ratio = exp(lograt);
	    double unif = uniform(rng);
	    bool accept = (unif < ratio) && isfinite(ratio);
	    return accept;
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

		logdensity_pop <pchi, dtheta> <<<nBlocks,nThreads>>>(Daug->GetDevChiPtr(), p_logdens_prop, p_logdens_function, ndata);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		double logdens_prop = thrust::reduce(d_proposed_logdens.begin(), d_proposed_logdens.end());

		logdens_prop += LogPrior(h_proposed_theta);

		// accept the proposed value?
		double metro_ratio = 0.0;
		bool accept = AcceptProp(logdens_prop, logdens_current, metro_ratio);
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

	void ResetAcceptance() { naccept = 0; }

	// setters and getters
	void SetDataAugPtr(boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > DataAug) {
		Daug = DataAug;
		ndata = Daug->GetDataDim();
		d_logdens.resize(ndata);
		d_proposed_logdens.resize(ndata);
	}

	void SetTheta(hvector& theta, bool update_logdens = true) {
		h_theta = theta;
	    double* p_theta = thrust::raw_pointer_cast(&theta[0]);
		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(c_theta, p_theta, dtheta * sizeof(*p_theta)));
		if (update_logdens) {
			// update value of conditional log-posterior for theta|chi
			double* p_chi = Daug->GetDevChiPtr(); // grab pointer to Daug.d_chi
			double* p_logdens = thrust::raw_pointer_cast(&d_logdens[0]);
			logdensity_pop <pchi, dtheta> <<<nBlocks,nThreads>>>(p_chi, p_logdens, p_logdens_function, ndata);
			CUDA_CHECK_RETURN(cudaPeekAtLastError());
			current_logdens = thrust::reduce(d_logdens.begin(), d_logdens.end());
		}
	}

	void SetLogDens(dvector& logdens) {
		d_logdens = logdens;
		current_logdens = thrust::reduce(logdens.begin(), logdens.end());
	}
	void SetCholFact(hvector cholfact_new) { cholfact = cholfact_new; }
	void SetCurrentIter(int iter) { current_iter = iter; }

	hvector GetHostTheta() { return h_theta; }

	hvector GetDevTheta() {
		// copy the current value of theta from constant GPU memory and return as a host-side vector
		double* p_theta;
		p_theta = GetDevThetaPtr();
		hvector copied_h_theta(dtheta);
		// c_theta array has 100 elements, but only need first dtheta elements
		for (int j = 0; j < dtheta; ++j) {
			copied_h_theta[j] = p_theta[j];
		}
		return copied_h_theta; }

	std::vector<double> GetTheta() { // return the current value of theta as a std::vector
		std::vector<double> std_theta(h_theta.size());
		thrust::copy(h_theta.begin(), h_theta.end(), std_theta.begin());
		return std_theta;
	}

	double GetLogDens() { return current_logdens; } // return the current value of summed log p(chi | theta);

	double* GetDevThetaPtr() {
		double* p_theta;
		// copy values of theta from __constant__ memory to host memory
		cudaMemcpyFromSymbol(p_theta, c_theta, sizeof(c_theta), 0, cudaMemcpyDeviceToHost);
		return p_theta;
	}

	hvector GetHostLogDens() {
		hvector h_logdens = d_logdens;
		return h_logdens;
	}
	dvector GetDevLogDens() { return d_logdens; }
	double* GetDevLogDensPtr() { return thrust::raw_pointer_cast(&d_logdens[0]); }
	int GetDim() { return dtheta; }
	hvector GetCholFactor() { return cholfact; }
	int GetNaccept() { return naccept; }
	pLogDensPop GetLogDensPopPtr() { return p_logdens_function; }
	boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > GetDataAugPtr() { return Daug; }

protected:
	// the value of the population parameter
	hvector h_theta;
	// log of the value the probability of the characteristics given the population parameter, chi | theta
	dvector d_logdens;
	dvector d_proposed_logdens;
	double current_logdens;
	// make sure that the population parameter knows about the characteristics
	boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > Daug;
	// cholesky factors of Metropolis proposal covariance matrix
	hvector cholfact;
	// interval variables used in robust adaptive metropolis algorithm
	hvector snorm_deviate;
	hvector scaled_proposal;
	// CUDA kernel launch specifications
	dim3& nBlocks;
	dim3& nThreads;
	// pointer to device-side function that compute the conditional log-posterior of characteristics|population
	pLogDensPop p_logdens_function;
	// MCMC parameters
	int naccept;
	int current_iter;
	int ndata;
};

#endif // _DATA_AUGMENTATION_HPP__ //
