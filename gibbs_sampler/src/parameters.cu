/*
 * parameters.cu
 *
 *  Created on: Jul 3, 2013
 *      Author: brandonkelly
 */

#import "include/parameters.cuh"

// constructor for data augmentation class that takes a std::vector of std::vectors as input
// for both the arrays of measurements and measurement uncertainties
DataAugmentation::DataAugmentation(vecvec& meas, vecvec& meas_unc)
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

	// grab pointers to the vectors
	// set initial values for the characteristics. this will launch a CUDA kernel.
	InitialValue();
}

// constructor for data augmentation class that takes pointers as inputs
DataAugmentation::DataAugmentation(double** meas, double** meas_unc, int n, int m) : ndata(n), mfeat(m)
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

	// set initial values for the characteristics. this will launch a CUDA kernel.
	InitialValue();
}

// set sizes of data members
void DataAugmentation::_SetArraySizes()
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

// set the initial values of the characteristics
__global__
void DataAugmentation::InitialValue(double* chi, double* meas, double* meas_unc, double* cholfact,
		double* logdens_meas, double* logdens_pop, int pchi)
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
}
