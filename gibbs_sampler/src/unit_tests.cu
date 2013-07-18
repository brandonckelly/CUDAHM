/*
 * unit_tests.cu
 *
 *  Created on: Jul 12, 2013
 *      Author: brandonkelly
 */

// standard includes
#include <iostream>
// local includes
#include "data_augmentation.cuh"

/* list of unit tests:
 *
 *	- test rank-1 cholesky update
 *	- make sure Chi::Propose follows a multivariate normal distribution
 *	- make sure Chi::Accept always accepts when the proposal and the current values are the same
 *	- make sure we accept and save a Chi value with a much higher posterior
 *	- Test Chi::Adapt acceptance rate and covariance by running a simple MCMC sampler
 *	- make sure PopulationPar::Propose follow a multivariate normal distribution
 *	- make sure that PopulationPar::Accept always accepts when the logdensities are the same
 *	- make sure PopulationPar::Update always accepts when the proposed and current theta values are the same
 *	- make sure we accept and save a PopulationPar value with the posterior is much higher
 *	- Test PopulationPar::Adapt acceptance rate and covariance by running a simple MCMC sampler
 *	- Test DataAugmentation::GetChi
 *	- make sure DataAugmentation::Update always accepts when the proposed and current chi values are the same
 *	- make sure we accept and save a Chi value when the posterior is much higher
 *
 */

int main(int argc, char** argv)
{

	int ndata = 10000;
	int mfeat = 4;
	int pchi = 3;
	int dtheta = 2;

	// Cuda grid launch
    dim3 nThreads(256);
    dim3 nBlocks((ndata + nThreads.x-1) / nThreads.x);
    printf("nBlocks: %d\n", nBlocks.x);  // no more than 64k blocks!
    if (nBlocks.x > 65535)
    {
        std::cerr << "ERROR: Block is too large" << std::endl;
    }

    double** meas;
    double** meas_unc;
    meas = new double* [ndata];
    meas_unc = new double* [ndata];
    for (int i = 0; i < ndata; ++i) {
		meas[i] = new double [mfeat];
		meas_unc[i] = new double [mfeat];
		for (int j = 0; j < mfeat; ++j) {
			meas[i][j] = 0.0;
			meas_unc[i][j] = 0.0;
		}
	}

	Characteristic Chi(pchi, mfeat, dtheta, 1);
	DataAugmentation<Characteristic> Daug(meas, meas_unc, ndata, mfeat, pchi, nBlocks, nThreads);
	PopulationPar<Characteristic> Theta(dtheta, Daug, nBlocks, nThreads);

	for (int i = 0; i < ndata; ++i) {
		delete [] meas[i];
		delete [] meas_unc[i];
	}
	delete meas;
	delete meas_unc;

	std::cout << "Success!!" << std::endl;
}
