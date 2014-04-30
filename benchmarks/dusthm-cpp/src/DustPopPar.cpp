//
//  DustPopPar.cpp
//  dusthm-cpp
//
//  Created by Brandon Kelly on 4/30/14.
//  Copyright (c) 2014 Brandon C. Kelly. All rights reserved.
//

#include <stdio.h>
#include <DustPopPar.hpp>

/*
 * Helper functions used by LogDensityPop to compute the log-density of log C, beta, log T | theta
 */

// calculate the inverse of a 3 x 3 matrix
double matrix_invert3d(double* A, double* A_inv) {
	double determ_inv = 0.0;
	determ_inv = 1.0 / (A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[8] * A[3] - A[5] * A[6]) +
                        A[2] * (A[3] * A[7] - A[4] * A[6]));
    
	A_inv[0] = determ_inv * (A[4] * A[8] - A[5] * A[7]);
	A_inv[1] = -determ_inv * (A[1] * A[8] - A[2] * A[7]);
	A_inv[2] = determ_inv * (A[1] * A[5]- A[2] * A[4]);
	A_inv[3] = -determ_inv * (A[3] * A[8] - A[5] * A[6]);
	A_inv[4] = determ_inv * (A[0] * A[8] - A[2] * A[6]);
	A_inv[5] = -determ_inv * (A[0] * A[5] - A[2] * A[3]);
	A_inv[6] = determ_inv * (A[3] * A[7] - A[4] * A[6]);
	A_inv[7] = -determ_inv * (A[0] * A[7] - A[1] * A[6]);
	A_inv[8] = determ_inv * (A[0] * A[4] - A[1] * A[3]);
    
	return determ_inv;
}

// calculate chisqr = transpose(x) * covar_inv * x
double chisqr(double* x, double* covar_inv, int nx)
{
	double chisqr = 0.0;
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < nx; ++j) {
			chisqr += x[i] * covar_inv[i * nx + j] * x[j];
		}
	}
	return chisqr;
}
