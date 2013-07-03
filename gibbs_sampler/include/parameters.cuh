/*
 * parameters.cuh
 *
 *  Created on: Jul 2, 2013
 *      Author: brandonkelly
 */

#ifndef PARAMETERS_CUH_
#define PARAMETERS_CUH_

// Standard includes
#include <cmath>
#include <vector>

// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

typedef std::vector<std::vector<double> > vecvec;

class PopulationPar; // forward declaration so that DataAugmentation knows about PopulationPar

// Base class for a data augmentation.
class DataAugmentation {
public:
	// Constructor when storing the measurements in std::vector
	DataAugmentation(vecvec& meas, vecvec& meas_unc);
	// Constructor when storing the measurements in arrays of pointers
	DataAugmentation(double** meas, double** meas_unc, int ndata, int mfeat);

	// Default Destructor
	~DataAugmentation();

	// calculate initial value of characteristics
	__global__ void virtual InitialValue();

	// make sure that the data augmentation knows about the population parameters
	void SetPopulation(PopulationPar& theta);

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
	__global__ void virtual Update();

	// methods to return the values of the characteristics and their log-densities
	vecvec GetChi();
	thrust::host_vector<double> GetLogDensPop();
	thrust::host_vector<double> GetLogDensMeas();

protected:
	// measurements and their uncertainties
	thrust::host_vector<double> h_meas;
	thrust::host_vector<double> h_meas_unc;
	thrust::device_vector<double> d_meas;
	thrust::device_vector<double> d_meas_unc;
	int ndata;
	int mfeat;
	// characteristics
	thrust::host_vector<double> h_chi;
	thrust::device_vector<double> d_chi;
	int pchi;
	// population-level parameters
	PopulationPar& theta;
	// logarithm of conditional posterior densities
	thrust::host_vector<double> h_logdens_meas; // probability of meas|chi
	thrust::device_vector<double> d_logdens_meas;
	thrust::host_vector<double> h_logdens_pop; // probability of chi|theta
	thrust::device_vector<double> d_logdens_pop;
	// cholesky factors of Metropolis proposal covariance matrix
	thrust::host_vector<double> h_cholfact;
	thrust::device_vector<double> d_cholfact;
};

// Base class for a population level parameter
class PopulationPar {
public:
	// constructor
	PopulationPar(DataAugmentation& daug);

	// Default destructor
	~PopulationPar();

	// calculate the initial value of the population parameters
	virtual void InitialValue();

	// return the log-prior of the population parameters
	virtual double LogPrior(thrust::host_vector<double> theta);

	// compute the conditional log-posterior density of the characteristics given the population parameter
	__global__ virtual double logdensity_pop(double* theta);

	// methods used to update the population parameters
	virtual double* Propose();
	virtual void AdaptProp();
	bool AcceptProp(double logdens_prop, double logdens_current, double forward_dens, double backward_dens);
	virtual void Update();

	// methods to set and return the value of the population parameter, theta
	thrust::host_vector<double> SetTheta();
	thrust::host_vector<double> GetTheta();

private:
	// the value of the population parameter
	thrust::host_vector<double> h_theta;
	thrust::device_vector<double> d_theta;
	// make sure that the population parameter know about the characteristics
	DataAugmentation& daug;
	// cholesky factors of Metropolis proposal covariance matrix
	thrust::host_vector<double> h_cholfact;
	thrust::device_vector<double> d_cholfact;
};

#endif /* PARAMETERS_CUH_ */
