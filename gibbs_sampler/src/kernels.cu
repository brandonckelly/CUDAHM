#include "kernels.cuh"

// Global random number generator and distributions for generating random numbers on the host. The random number generator used
// is the Mersenne Twister mt19937 from the BOOST library. These are instantiated in data_augmentation.cu.
extern boost::random::mt19937 rng;
extern boost::random::normal_distribution<> snorm; // Standard normal distribution
extern boost::random::uniform_real_distribution<> uniform; // Uniform distribution from 0.0 to 1.0

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

// calculate initial value of characteristics
__global__
void initial_chi_value(double* chi, double* meas, double* meas_unc, double* cholfact, double* logdens,
		int ndata, int mfeat, int pchi, pLogDensMeas LogDensityMeas)
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
		// copy value to registers
		double this_chi[3];
		for (int j = 0; j < pchi; ++j) {
			this_chi[j] = chi[j * ndata + idata];
		}
		logdens[idata] = LogDensityMeas(this_chi, meas, meas_unc, pchi, mfeat);
	}
}

// kernel to update the values of the characteristics in parallel on the GPU
__global__
void update_characteristic(double* meas, double* meas_unc, double* chi, double* theta, double* cholfact,
		double* logdens_meas, double* logdens_pop, curandState* devStates, pLogDensMeas LogDensityMeas,
		pLogDensPop LogDensityPop, int current_iter, int* naccept, int ndata, int mfeat, int pchi, int dim_theta)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		curandState localState = devStates[idata]; // grab state of this random number generator

		// copy values for this data point to registers for speed
		// TODO: convert these arrays to shared memory
		double snorm_deviate[3], scaled_proposal[3], proposed_chi[3], local_chi[3], local_cholfact[6];
		int cholfact_index = 0;
		for (int j = 0; j < pchi; ++j) {
			local_chi[j] = chi[j * ndata + idata];
			for (int k = 0; k < (j+1); ++k) {
				local_cholfact[cholfact_index] = cholfact[cholfact_index * ndata + idata];
				cholfact_index++;
			}
		}

		// propose a new value of chi
		Propose(local_chi, local_cholfact, proposed_chi, snorm_deviate, scaled_proposal, pchi, &localState);

		// get value of log-posterior for proposed chi value
		double logdens_meas_prop = LogDensityMeas(proposed_chi, meas, meas_unc, mfeat, pchi);
		double logdens_pop_prop = LogDensityPop(proposed_chi, theta, pchi, dim_theta);
		double logpost_prop = logdens_meas_prop + logdens_pop_prop;

		// accept the proposed value of the characteristic?
		double logpost_current = logdens_meas[idata] + logdens_pop[idata];
		double metro_ratio = 0.0;
		bool accept = AcceptProp(logpost_prop, logpost_current, 0.0, 0.0, metro_ratio, &localState);

		// adapt the covariance matrix of the characteristic proposal distribution
		AdaptProp(local_cholfact, snorm_deviate, scaled_proposal, metro_ratio, pchi, current_iter);

		int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
		for (int j = 0; j < dim_cholfact; ++j) {
			// copy value of this adapted cholesky factor back to global memory
			cholfact[j * ndata + idata] = local_cholfact[j];
		}
		current_iter++;

		// copy local RNG state back to global memory
		devStates[idata] = localState;

		if (accept) {
			// accepted this proposal, so save new value of chi and log-densities
			for (int j=0; j<pchi; j++) {
				chi[ndata * j + idata] = proposed_chi[j];
			}
			logdens_meas[idata] = logdens_meas_prop;
			logdens_pop[idata] = logdens_pop_prop;
			naccept[idata] += 1;
		}
	}

}

// compute the conditional log-posterior density of the characteristics given the population parameter
__global__
void logdensity_meas(double* meas, double* meas_unc, double* chi, double* logdens, pLogDensMeas LogDensityMeas,
		int ndata, int mfeat, int pchi)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		double chi_i[3];
		for (int j = 0; j < pchi; ++j) {
			chi_i[j] = chi[j * ndata + idata];
		}
		logdens[idata] = LogDensityMeas(chi_i, meas, meas_unc, mfeat, pchi);
	}
}

// compute the conditional log-posterior density of the characteristics given the population parameter
__global__
void logdensity_pop(double* theta, double* chi, double* logdens, pLogDensPop LogDensityPop, int ndata,
		int pchi, int dim_theta)
{
	int idata = blockDim.x * blockIdx.x + threadIdx.x;
	if (idata < ndata)
	{
		double chi_i[3];
		for (int j = 0; j < pchi; ++j) {
			chi_i[j] = chi[j * ndata + idata];
		}
		logdens[idata] = LogDensityPop(chi_i, theta, pchi, dim_theta);
	}
}



