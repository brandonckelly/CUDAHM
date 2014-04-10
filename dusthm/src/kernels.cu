#include "kernels.cuh"

// Global random number generator and distributions for generating random numbers on the host. The random number generator used
// is the Mersenne Twister mt19937 from the BOOST library.
boost::random::mt19937 rng;
boost::random::normal_distribution<> snorm(0.0, 1.0); // Standard normal distribution
boost::random::uniform_real_distribution<> uniform(0.0, 1.0); // Uniform distribution from 0.0 to 1.0

__constant__ double c_theta[100];

// Function to compute the rank-1 Cholesky update/downdate. Note that this is done in place.
__device__ __host__
void chol_update_r1(double* cholfactor, double* v, int dim_v, bool downdate) {

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

// compute the conditional log-posterior density of the measurements given the characteristic
//__device__ double LogDensityMeas(double* chi, double* meas, double* meas_unc, int mfeat, int pchi) { return 0.0; }

// compute the conditional log-posterior density of the characteristic given the population parameter
//__device__ double LogDensityPop(double* chi, double* theta, int pchi, int dim_theta) { return 0.0; }

// propose a new value for the characteristic
__device__ __host__
void Propose(double* chi, double* cholfact, double* proposed_chi, double* snorm_deviate,
		double* scaled_proposal, int pchi, curandState* p_state)
{
	// get the unit proposal
	for (int j=0; j<pchi; j++) {
#ifdef __CUDA_ARCH__
		snorm_deviate[j] = curand_normal_double(p_state);
#else
		snorm_deviate[j] = snorm(rng);
#endif
	}

	// propose a new chi value
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
}

// adapt the covariance matrix of the proposals for the characteristics
__device__ __host__
void AdaptProp(double* cholfact, double* snorm_deviate, double* scaled_proposal, double metro_ratio,
		int pchi, int current_iter)
{
	double unit_norm = 0.0;
	for (int j=0; j<pchi; j++) {
		unit_norm += snorm_deviate[j] * snorm_deviate[j];
	}
	unit_norm = sqrt(unit_norm);
	double decay_rate = 0.667;
	double target_rate = 0.4;
	double decay_sequence = 1.0 / pow(current_iter, decay_rate);
	double scaled_coef = sqrt(decay_sequence * fabs(metro_ratio - target_rate)) / unit_norm;
	for (int j=0; j<pchi; j++) {
		scaled_proposal[j] *= scaled_coef;
	}
	bool downdate = (metro_ratio < target_rate);
	// do rank-1 cholesky update to update the proposal covariance matrix
	chol_update_r1(cholfact, scaled_proposal, pchi, downdate);
}

// decide whether to accept or reject the proposal based on the metropolist-hasting ratio
__device__ __host__
bool AcceptProp(double logdens_prop, double logdens_current, double forward_dens,
		double backward_dens, double& ratio, curandState* p_state)
{
	double lograt = logdens_prop - forward_dens - (logdens_current - backward_dens);
	lograt = min(lograt, 0.0);
	ratio = exp(lograt);
#ifdef __CUDA_ARCH__
	double unif = curand_uniform_double(p_state);
#else
	double unif = uniform(rng);
#endif
	bool accept = (unif < ratio) && isfinite(ratio);
	return accept;
}

/*
 * Kernels
 */

// Initialize the parallel random number generator state on the device
__global__ void initialize_rng(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
     number, no offset */
    curand_init(1234, id, 0, &state[id]);
}
