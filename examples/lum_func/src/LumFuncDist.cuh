/*
* LumFuncDist.cuh
*
*  Created on: December 21, 2014
*      Author: Janos M. Szalai-Gindl
*/

#ifndef LUMFUNCDIST_CUH_
#define LUMFUNCDIST_CUH_

// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class LumFuncDist
{
public:
	LumFuncDist(thrust::device_vector<double>& d_distData) : d_distData(d_distData)
	{
	}

	double* GetDistData() { return thrust::raw_pointer_cast(&d_distData[0]); }

protected:
	thrust::device_vector<double> d_distData;
};

#endif /* LUMFUNCDIST_CUH_ */