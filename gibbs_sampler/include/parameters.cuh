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

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

typedef std::vector<std::vector<double> > vecvec;
typedef hvector hvector;
typedef dvector dvector;

// Global random number generator and distributions for generating random numbers on the host. The random number generator used
// is the Mersenne Twister mt19937 from the BOOST library.
boost::random::mt19937 rng;
boost::random::normal_distribution<> snorm(0.0, 1.0); // Standard normal distribution
boost::random::uniform_real_distribution<> uniform(0.0, 1.0); // Uniform distribution from 0.0 to 1.0

// constant memory contains array sizes and MCMC sampler parameters
__constant__ int c_ndata; // # of data points
__constant__ int c_mfeat; // # of measured features per data point
__constant__ int c_pchi; // # of characteristics per data point
__constant__ int c_dim_theta; // # dimension of population parameters
__constant__ double c_target_rate; // MCMC sampler target acceptance rate
__constant__ double c_decay_rate; // decay rate of robust adaptive metropolis algorithm

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
	// Constructor
	DataAugmentation(double** meas, double** meas_unc, int n, int m, dim3& nB, dim3& nT) :
		ndata(n), mfeat(m), nBlocks(nB), nThreads(nT)
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

		// place array sizes constant memory
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(&c_ndata, &ndata, sizeof(ndata)));
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(&c_mfeat, &mfeat, sizeof(ndata)));
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(&c_pchi, &pchi, sizeof(ndata)));

		thrust::fill(h_cholfact.begin(), h_cholfact.end(), 0.0);
		d_cholfact = h_cholfact;

		// grab pointers to the device vector memory locations
		double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
		double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
		double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
		double* p_cholfact = thrust::raw_pointer_cast(&d_cholfact[0]);
		double* p_logdens_meas = thrust::raw_pointer_cast(&d_logdens_meas[0]);
		double* p_logdens_pop = thrust::raw_pointer_cast(&d_logdens_pop[0]);

		// Allocate memory on GPU for RNG states
		CUDA_CHECK_RETURN(cudaMalloc((void **)&p_devStates, nThreads.x * nBlocks.x * sizeof(curandState)));
		// Initialize the random number generator states on the GPU
		InitializeRNG<<<nBlocks,nThreads>>>(p_devStates);

		// Wait until RNG stuff is done running on the GPU, make sure everything went OK
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		// set initial values for the characteristics. this will launch a CUDA kernel.
		InitialValue<ChiType><<<nBlocks,nThreads>>>(p_chi, p_meas, p_meas_unc, p_cholfact, p_logdens_meas,
				p_logdens_pop, ndata, mfeat, pchi);

		// copy values from device to host
		h_chi = d_chi;
		h_cholfact = d_cholfact;
		h_logdens_meas = d_logdens_meas;
		h_logdens_pop = d_logdens_pop;
	}

	virtual ~DataAugmentation() {
		cudaFree(p_devStates);
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

	// Initialize the parallel random number generator state on the device
	__global__ void InitializeRNG(curandState *state)
	{
	    int id = threadIdx.x + blockIdx.x * blockDim.x;
	    /* Each thread gets same seed, a different sequence
	     number, no offset */
	    curand_init(1234, id, 0, &state[id]);
	}

	// make sure that the data augmentation knows about the population parameters
	void SetPopulation(PopulationPar& t) {
		theta = t;
	}

	// launch the update kernel on the GPU
	void Update()
	{
		// grab the pointers to the device memory locations
		double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
		double* p_meas = thrust::raw_pointer_cast(&d_meas[0]);
		double* p_meas_unc = thrust::raw_pointer_cast(&d_meas_unc[0]);
		double* p_cholfact = thrust::raw_pointer_cast(&d_cholfact[0]);
		double* p_logdens_meas = thrust::raw_pointer_cast(&d_logdens_meas[0]);
		double* p_logdens_pop = thrust::raw_pointer_cast(&d_logdens_pop[0]);
		int* p_naccept = thrust::raw_pointer_cast(&d_naccept[0]);

		// launch the kernel to update the characteristics on the GPU
		UpdateKernel<ChiType><<nBlocks,nThreads>>>(p_meas, p_meas_unc, p_chi, p_cholfact, p_logdens_meas, p_logdens_pop,
				p_devStates, current_iter, p_naccept);

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        // transfer values back to host
        h_chi = d_chi;
        h_logdens_pop = d_logdens_pop;
        h_logdens_meas = d_logdens_meas;
        h_naccept = d_naccept;

        current_iter++;
	}

	// kernel to update the values of the characteristics in parallel on the GPU
	template <class ChiType> __global__
	virtual void UpdateKernel(double* meas, double* meas_unc, double* chi, double* cholfact,
			double* logdens_meas, double* logdens_pop, curandState* devStates, int current_iter, int* naccept)
	{
		// TODO: place ndata, pchi, mfeat, target_rate, and decay_rate into constant memory


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
	// state of parallel random number generator on the device
	curandState* p_devStates;
	// CUDA kernel launch specifications
	dim3& nBlocks;
	dim3& nThreads;
	// MCMC sampler parameters
	int current_iter;
	double target_rate;
	double decay_rate;
	thrust::host_vector<int> h_naccept;
	thrust::device_vector<int> d_naccept;
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

		// place array size constant memory
        CUDA_CHECK_RETURN(cudaMemcpyToSymbol(&c_dim_theta, &dim_theta, sizeof(dim_theta)));

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

// Base class for an individual data point's characteristic, i.e., chi_i
class Characteristic {
public:
	// constructor
	__device__ __host__ Characteristic(curandState& localState, int iter, double rate) :
	state(localState), current_iter(iter), target_rate(rate), decay_rate(0.667);
	__device__ __host__ virtual ~Characteristic() {}

	// compute the conditional log-posterior density of the measurements given the characteristic
	__device__ __host__ virtual double logdensity_meas(double* chi, double* meas, double* meas_unc, int pchi, int mfeat)
	{
		return 0.0;
	}

	// compute the conditional log-posterior dentity of the characteristic given the population parameter
	__device__ __host__ virtual double logdensity_pop(double* chi, double* theta, int pchi, int dim_theta)
	{
		return 0.0;
	}

	// propose a new value for the characteristic
	__device__ __host__ virtual double* Propose(double* chi, double* cholfact, double* snorm_deviate, double* scaled_proposal, int pchi)
	{
		// get the unit proposal
		for (int j=0; j<pchi; j++) {
			snorm_deviate[j] = curand_normal_double(&state);
		}

		// propose a new chi value
		double proposed_chi[pchi];
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
		return proposed_chi;
	}

	// adapt the covariance matrix of the proposals for the characteristics
	__device__ __host__ virtual void AdaptProp(double* cholfact, double* snorm_deviate, double* scaled_proposal,
			double metro_ratio, int pchi)
	{
		double unit_norm = 0.0;
		for (int j=0; j<pchi; j++) {
			unit_norm += snorm_deviate[j] * snorm_deviate[j];
		}
		unit_norm = sqrt(unit_norm);
		double decay_sequence = 1.0 / pow((double) current_iter, decay_rate);
		double scaled_coef = sqrt(decay_sequence * fabs(metro_ratio - target_rate)) / unit_norm;
		for (int j=0; j<pchi; j++) {
			scaled_proposal[j] *= scaled_coef;
		}
		bool downdate = (metro_ratio < target_rate);
		// do rank-1 cholesky update to update the proposal covariance matrix
		CholUpdateR1(cholfact, scaled_proposal, pchi, downdate);
	}

	// decide whether to accept or reject the proposal based on the metropolist-hasting ratio
	__device__ __host__ bool AcceptProp(double logdens_prop, double logdens_current, double forward_dens,
			double backward_dens)
	{
		double lograt = logdens_prop - forward_dens - (logdens_current - backward_dens);
		lograt = min(lograt, 0.0);
		double ratio = exp(lograt);
		double unif = curand_uniform_double(&state);
		bool accept = (unif < ratio) && std::isfinite(ratio);
		return accept;
	}

protected:
	// random number generator state for this characteristic
	curandState& state;
	// MCMC sampler parameters
	int current_iter;
	double decay_rate;
	double target_rate;
};

#endif /* PARAMETERS_CUH_ */
