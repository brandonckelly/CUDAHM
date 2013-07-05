/*
 * parameters.cuh
 *
 *  Created on: Jul 2, 2013
 *      Author: brandonkelly
 */

#ifndef PARAMETERS_CUH_
#define PARAMETERS_CUH_

// Cuda Includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Standard includes
#include <cmath>
#include <vector>
// Boost includes
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

typedef std::vector<std::vector<double> > vecvec;
typedef hvector hvector;
typedef dvector dvector;

// Global random number generator and distributions for generating random numbers on the host. The random number generator used
// is the Mersenne Twister mt19937 from the BOOST library.
boost::random::mt19937 rng;
boost::random::normal_distribution<> snorm(0.0, 1.0); // Standard normal distribution
boost::random::uniform_real_distribution<> uniform(0.0, 1.0); // Uniform distribution from 0.0 to 1.0

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

template <class ChiType>
class PopulationPar; // forward declaration so that DataAugmentation knows about PopulationPar

// Base class for a data augmentation.
template <class ChiType>
class DataAugmentation {

private:
	// dimension of characteristics vector. this must be explicitly added by derived classes
	int pchi = 2;

public:
	// Constructor when storing the measurements in std::vector
	DataAugmentation(vecvec& meas, vecvec& meas_unc, dim3& nB, dim3& nT) : nBlocks(nB), nThreads(nT)
	{
		int size1 = meas.size();
		int size2 = meas[0].size();

		ndata = max(size1,size2); // assume that ndata < mfeat and figure out the values from the size
		mfeat = min(size1,size2); // of the arrays.

		_SetArraySizes();

		// copy input data to data members
		for (int j = 0; j < mfeat; ++j) {
			for (int i = 0; i < ndata; ++i) {
				if (size1 < size2) {
					h_meas[ndata * j + i] = meas[j][i];
					h_meas_unc[ndata * j + i] = meas_unc[j][i];
				} else {
					h_meas[ndata * j + i] = meas[i][j];
					h_meas_unc[ndata * j + i] = meas_unc[i][j];
				}
			}
		}
		// copy data from host to device
		d_meas = h_meas;
		d_meas_unc = h_meas_unc;

		thrust::fill(h_cholfact.begin(), h_cholfact.end(), 0.0);
		d_cholfact = h_cholfact;

		// grab pointers to the device vector memory locations
		double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
		double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
		double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
		double* p_cholfact = thrust::raw_pointer_cast(&d_cholfact[0]);
		double* p_logdens_meas = thrust::raw_pointer_cast(&d_logdens_meas[0]);
		double* p_logdens_pop = thrust::raw_pointer_cast(&d_logdens_pop[0]);

		// set initial values for the characteristics. this will launch a CUDA kernel.
		InitialValue<ChiType><<<nBlocks,nThreads>>>(p_chi, p_meas, p_meas_unc, p_cholfact, p_logdens_meas,
				p_logdens_pop, ndata, mfeat, pchi);

		// copy values from device to host
		h_chi = d_chi;
		h_cholfact = d_cholfact;
		h_logdens_meas = d_logdens_meas;
		h_logdens_pop = d_logdens_pop;
	}

	// Constructor when storing the measurements in arrays of pointers
	DataAugmentation(double** meas, double** meas_unc, int n, int m, dim3& nB, dim3& nT) : nBlocks(nB), nThreads(nT)
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

		thrust::fill(h_cholfact.begin(), h_cholfact.end(), 0.0);
		d_cholfact = h_cholfact;

		// grab pointers to the device vector memory locations
		double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
		double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
		double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
		double* p_cholfact = thrust::raw_pointer_cast(&d_cholfact[0]);
		double* p_logdens_meas = thrust::raw_pointer_cast(&d_logdens_meas[0]);
		double* p_logdens_pop = thrust::raw_pointer_cast(&d_logdens_pop[0]);

		// set initial values for the characteristics. this will launch a CUDA kernel.
		InitialValue<ChiType><<<nBlocks,nThreads>>>(p_chi, p_meas, p_meas_unc, p_cholfact, p_logdens_meas,
				p_logdens_pop, ndata, mfeat, pchi);

		// copy values from device to host
		h_chi = d_chi;
		h_cholfact = d_cholfact;
		h_logdens_meas = d_logdens_meas;
		h_logdens_pop = d_logdens_pop;
	}

	// calculate initial value of characteristics
	template <class ChiType> __global__
	void virtual InitialValue(double* chi, double* meas, double* meas_unc, double* cholfact,
			double* logdens_meas, double* logdens_pop, int ndata, int mfeat, int pchi)
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
		}
		// calculate initial log-posteriors
	}

	// make sure that the data augmentation knows about the population parameters
	void SetPopulation(PopulationPar& t) {
		theta = t;
	}

	__global__
	void virtual Update()
	{

	}

	// methods to return the values of the characteristics and their log-densities
	vecvec GetChi()
	{
		vecvec chi(ndata);
		// grab values of characteristics from host vector
		for (int i = 0; i < ndata; ++i) {
			std::vector<double> chi_i(pchi);
			for (int j = 0; j < pchi; ++j) {
				chi_i[j] = h_chi[ndata * j + i];
			}
			chi[i] = chi_i;
		}
		return chi;
	}

	hvector GetLogDensPop() {
		return h_logdens_pop;
	}
	hvector GetLogDensMeas() {
		return h_logdens_meas;
	}

protected:
	// set the sizes of the data members
	void _SetArraySizes()
	{
		h_meas.resize(ndata * mfeat);
		d_meas.resize(ndata * mfeat);
		h_meas_unc.resize(ndata * mfeat);
		d_meas_unc.resize(ndata * mfeat);
		h_logdens_meas.resize(ndata);
		d_logdens_meas.resize(ndata);
		h_logdens_pop.resize(ndata);
		d_logdens_pop.resize(ndata);
		h_chi.resize(ndata * pchi);
		d_chi.resize(ndata * pchi);
		int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
		h_cholfact.resize(ndata * dim_cholfact);
		d_cholfact.resize(ndata * dim_cholfact);
	}

	// measurements and their uncertainties
	hvector h_meas;
	hvector h_meas_unc;
	dvector d_meas;
	dvector d_meas_unc;
	int ndata;
	int mfeat;
	// characteristics
	hvector h_chi;
	dvector d_chi;
	// population-level parameters
	PopulationPar<ChiType>& theta;
	// logarithm of conditional posterior densities
	hvector h_logdens_meas; // probability of meas|chi
	dvector d_logdens_meas;
	hvector h_logdens_pop; // probability of chi|theta
	dvector d_logdens_pop;
	// cholesky factors of Metropolis proposal covariance matrix
	hvector h_cholfact;
	dvector d_cholfact;
	// CUDA kernel launch specifications
	dim3& nBlocks;
	dim3& nThreads;
};

// Base class for a population level parameter
template <class ChiType>
class PopulationPar {

private:
	// dimension of the population parameters. this must be explicitly set in derived classes.
	int dim_theta = 2;

public:
	// constructor
	PopulationPar(double rate, DataAugmentation<ChiType>& D, dim3& nB, dim3& nT) :
		target_rate(rate), daug(D), nBlocks(nB), nThreads(nT)
	{
		h_theta.resize(dim_theta);
		d_theta = h_theta;
		int dim_cholfact = dim_theta * dim_theta - ((dim_theta - 1) * dim_theta) / 2;
		cholfact.resize(dim_cholfact);

		decay_rate = 2.0 / 3.0;

		InitialValue();
	}

	// calculate the initial value of the population parameters
	virtual void InitialValue()
	{
		// set initial value of theta to zero
		thrust::fill(h_theta.begin(), h_theta.end(), 0.0);
		d_theta = h_theta;

		// set initial covariance matrix of the theta proposals as the identity matrix
		thrust::fill(cholfact.begin(), cholfact.end(), 0.0);
		int diag_index = 0;
		for (int k=0; k<dim_theta; k++) {
			cholfact[diag_index] = 1.0;
			diag_index += k + 2;
		}

		// get initial value of conditional log-posterior for theta|chi


		// reset the number of MCMC iterations
		current_iter = 1;
	}

	// return the log-prior of the population parameters
	virtual double LogPrior(hvector theta) {
		return 0.0;
	}

	// compute the conditional log-posterior density of the characteristics given the population parameter
	template<class ChiType>
	__global__ virtual double logdensity_pop(double* theta) {
		return 0.0;
	}

	// propose a new value of the population parameters
	virtual hvector Propose(hvector& snorm_deviate, hvector& scaled_proposal)
	{
        // get the unit proposal
        for (int k=0; k<dim_theta; k++) {
            snorm_deviate[k] = snorm(rng);
        }

        // transform unit proposal so that is has a multivariate normal distribution
        hvector proposed_theta;
        thrust::fill(scaled_proposal.begin(), scaled_proposal.end(), 0.0);
        int cholfact_index = 0;
        for (int j=0; j<dim_theta; j++) {
            for (int k=0; k<(j+1); k++) {
                scaled_proposal[j] += cholfact[cholfact_index] * snorm_deviate[k];
                cholfact_index++;
            }
            proposed_theta[j] = h_theta[j] + scaled_proposal[j];
        }

        return proposed_theta;
	}

	// adapt the covariance matrix (i.e., the cholesky factors) of the theta proposals
	virtual void AdaptProp(hvector& snorm_deviate, hvector& scaled_proposal, double metro_ratio)
	{
		double unit_norm = 0.0;
	    for (int j=0; j<dim_theta; j++) {
	    	unit_norm += snorm_deviate[j] * snorm_deviate[j];
	    }
        unit_norm = sqrt(unit_norm);
        double decay_sequence = 1.0 / pow(current_iter, decay_rate);
        double scaled_coef = sqrt(decay_sequence * fabs(metro_ratio - target_rate)) / unit_norm;
        for (int j=0; j<dim_theta; j++) {
            scaled_proposal[j] *= scaled_coef;
        }

        bool downdate = (metro_ratio < target_rate);
        double* p_cholfact = thrust::raw_pointer_cast(&cholfact[0]);
        double* p_scaled_proposal = thrust::raw_pointer_cast(&scaled_proposal[0]);
        // rank-1 update of the cholesky factor
        CholUpdateR1(p_cholfact, p_scaled_proposal, dim_theta, downdate);
	}

	bool AcceptProp(double logdens_prop, double logdens_current, double forward_dens = 0.0, double backward_dens = 0.0)
	{
        double lograt = logdens_prop - forward_dens - (logdens_current - backward_dens);
        lograt = std::min(lograt, 0.0);
        double ratio = exp(lograt);
        double unif = uniform(rng);
        bool accept = (unif < ratio) && std::isfinite(ratio);
        return accept;
	}

	virtual void Update();

	// methods to set and return the value of the population parameter, theta
	void SetTheta(hvector theta) {
		h_theta = theta;
		d_theta = h_theta;
	}
	hvector GetTheta() {
		return h_theta;
	}

protected:
	// the value of the population parameter
	hvector h_theta;
	dvector d_theta;
	// make sure that the population parameter knows about the characteristics
	DataAugmentation<ChiType>& daug;
	// cholesky factors of Metropolis proposal covariance matrix
	hvector cholfact;
	// CUDA kernel launch specifications
	dim3& nBlocks;
	dim3& nThreads;
	// MCMC parameters
	double target_rate; // target acceptance rate for metropolis algorithm
	double decay_rate; // decay rate for robust metropolis algorithm, gamma in notation of Vihola (2012)
	int current_iter;
};

class Characteristic {
public:

	// grab the chi, meas, and meas_unc array from global memory for data point index tid.
	__device__ double* GrabGlobalChi(int tid);
	__device__ double* GrabGlobalMeas(int tid);
	__device__ double* GrabGlobalUnc(int tid);

	// methods to compute the conditional log-posterior densities
	__device__ __host__ virtual double logdensity_meas(double* chi) = 0;
	__device__ __host__ virtual double logdensity_pop(double* chi) = 0;

	// methods used to update the characteristics
	__device__ __host__ virtual double* Propose();
	__device__ __host__ virtual void AdaptProp();
	__device__ __host__ bool AcceptProp(double logdens_prop, double logdens_current, double forward_dens,
			double backward_dens);

private:
};

#endif /* PARAMETERS_CUH_ */
