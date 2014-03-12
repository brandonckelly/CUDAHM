/*
 * normnorm.cpp
 *
 *  Created on: Mar 12, 2014
 *      Author: brandonkelly
 */

// local includes
#include "../src/GibbsSampler.hpp"

/*
 * Pointers to the device-side functions used to compute the conditional log-posteriors
 */
__constant__ pLogDensMeas c_LogDensMeas = LogDensityMeas;
__constant__ pLogDensPop c_LogDensPop = LogDensityPop;

extern __constant__ double c_theta[100];

// compute the conditional log-posterior density of the measurements given the characteristic
__device__ __host__
double LogDensityMeas(double* chi, double* meas, double* meas_unc, int mfeat, int pchi)
{
	double logdens = 0.0;
	for (int i = 0; i < pchi; ++i) {
		double chi_std = (meas[i] - chi[i]) / meas_unc[i];
		logdens += -0.5 * chi_std * chi_std;
	}

	return logdens;
}

// compute the conditional log-posterior density of the characteristic given the population mean
__device__ __host__
double LogDensityPop(double* chi, double* theta, int pchi, int dim_theta)
{
	// known inverse covariance matrix of the characteristics
	double covar_inv[9] =
	{
			0.64880351, -2.66823952, 0.10406763,
			-2.66823952, 17.94430089, -0.55439855,
			0.10406763, -0.55439855, 0.02455399
	};
	// subtract off the population mean
	double chi_cent[3];
	for (int i = 0; i < 3; ++i) {
		chi_cent[i] = chi[i] - theta[i];
	}

	double logdens = -0.5 * ChiSqr(chi_cent, covar_inv, 3);
	return logdens;
}




int main(int argc, char** argv)
{

}
