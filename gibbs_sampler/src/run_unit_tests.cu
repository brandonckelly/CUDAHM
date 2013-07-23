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
#include "UnitTests.cuh"

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
 *	- make sure we get the correct acceptance rate and covariance of the characteristics by keeping the population
 *	parameter fixed
 *
 */

int main(int argc, char** argv)
{
	int ndata = 10000;
	int mfeat = 3;
	int pchi = 3;
	int dtheta = 3;

	// Cuda grid launch
    dim3 nThreads(256);
    dim3 nBlocks((ndata + nThreads.x-1) / nThreads.x);
    printf("nBlocks: %d\n", nBlocks.x);  // no more than 64k blocks!
    if (nBlocks.x > 65535)
    {
        std::cerr << "ERROR: Block is too large" << std::endl;
        return 2;
    }

    /*
    double** meas_temp;
    double** meas_unc_temp;
	// fill data arrays
    meas_temp = new double* [ndata];
    meas_unc_temp = new double* [ndata];
    for (int i = 0; i < ndata; ++i) {
		meas_temp[i] = new double [mfeat];
		meas_unc_temp[i] = new double [mfeat];
		for (int j = 0; j < mfeat; ++j) {
			meas_temp[i][j] = 0.0;
			meas_unc_temp[i][j] = 0.0;
		}
	}

    DataAugmentation<Characteristic> Daug(meas_temp, meas_unc_temp, ndata, mfeat, pchi, nBlocks, nThreads);
    PopulationPar<Characteristic> Theta(dtheta, &Daug, nBlocks, nThreads);
    Characteristic Chi(pchi, mfeat, dtheta, 1);

	*/

    UnitTests Tests(ndata, mfeat, pchi, dtheta, nBlocks, nThreads);

    // test the rank-1 cholesky update
    Tests.R1CholUpdate();

    // tests for the characteristic class
    Tests.ChiPropose();
    Tests.ChiAcceptSame();
    Tests.ChiAdapt();

    // tests for population parameter class
    Tests.ThetaPropose();
    Tests.ThetaAcceptSame();
    Tests.ThetaAdapt();

    // tests for the data augmentation class
    Tests.DaugPopPtr();

    // print results
    Tests.Finish();

    /*
	for (int i = 0; i < ndata; ++i) {
		delete [] meas_temp[i];
		delete [] meas_unc_temp[i];
	}
	delete meas_temp;
	delete meas_unc_temp;
     */

	std::cout << "Success!!" << std::endl;
}
