//
//  DataAugmentation.hpp
//  dusthm-cpp
//
//  Created by Brandon Kelly on 4/30/14.
//  Copyright (c) 2014 Brandon C. Kelly. All rights reserved.
//

#ifndef dusthm_cpp_DataAugmentation_hpp
#define dusthm_cpp_DataAugmentation_hpp

/*
 * parameters.cuh
 *
 *  Created on: Jul 28, 2013
 *      Author: brandonkelly
 */

#ifndef _DATA_AUGMENTATION_HPP__
#define _DATA_AUGMENTATION_HPP__

// Standard includes
#include <cmath>
#include <vector>
#include <stdio.h>
#include <numeric>

// Boost includes
#include <boost/shared_ptr.hpp>

// Local includes
#include "chol_update_r1.hpp"

// Global random number generator and distributions for generating random numbers on the host. The random number generator used
// is the Mersenne Twister mt19937 from the BOOST library.
extern boost::random::mt19937 rng;
extern boost::random::normal_distribution<> snorm; // Standard normal distribution
extern boost::random::uniform_real_distribution<> uniform; // Uniform distribution from 0.0 to 1.0

// Global constants for MCMC sampler
const double target_rate = 0.4; // MCMC sampler target acceptance rate
const double decay_rate = 0.667; // decay rate of robust adaptive metropolis algorithm

// convenience typedefs
typedef std::vector<std::vector<double> > vecvec;
typedef std::vector<double> svector;

template <int mfeat, int pchi, int dtheta> class PopulationPar; // forward declaration

// class for a data augmentation.
template <int mfeat, int pchi, int dtheta>
class DataAugmentation
{
public:
	// Constructor
	DataAugmentation(vecvec& meas, vecvec& meas_unc) : ndata(meas.size()) {
		// set sizes of arrays
		meas.resize(ndata * mfeat);
		meas_unc.resize(ndata * mfeat);
		logdens.resize(ndata);
		chi.resize(ndata * pchi);
		int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
		cholfact.resize(ndata * dim_cholfact);
		naccept.resize(ndata);
        
		// copy input data to data members
		for (int j = 0; j < mfeat; ++j) {
			for (int i = 0; i < ndata; ++i) {
				meas[ndata * j + i] = meas[i][j];
				meas_unc[ndata * j + i] = meas_unc[i][j];
			}
		}
        
		thrust::fill(cholfact.begin(), cholfact.end(), 0.0);
        
        save_trace = true;
	}
 
    virtual void LogDensity(double* chi, double* meas, double* meas_unc) {return 0.0;}
    
	virtual void Initialize() {
        // initialize all chi values to zero
        chi.fill(chi.begin(), chi.end(), 0.0)

        // set initial covariance matrix of the chi proposals as the identity matrix
        for (int i=0; i<ndata; i++) {
            int diag_index = 0;
            for (int j=0; j<pchi; j++) {
                cholfact[idata + ndata * diag_index] = 1.0;
                diag_index += j + 2;  // cholesky factor is lower diagonal
            }
        }
        
        // set initial value of the log p(meas|chi)
        logdens = LogDensity(chi, meas, meas_unc);
        
		thrust::fill(naccept.begin(), naccept.end(), 0);
		current_iter = 1;
	}
    
	// launch the update kernel on the GPU
	virtual void Update() {

        svector logdens_pop = p_Theta->GetLogDens()
        
        for (int i=0; i<ndata; i++) {
            // copy values for this data point for consistency with CUDAHM
            double snorm_deviate[pchi], scaled_proposal[pchi], proposed_chi[pchi], local_chi[pchi];
            const int dim_cholfact = pchi * pchi - ((pchi - 1) * pchi) / 2;
            double local_cholfact[dim_cholfact];
            int cholfact_index = 0;
            for (int j = 0; j < pchi; ++j) {
                local_chi[j] = chi[j * ndata + i];
                for (int k = 0; k < (j+1); ++k) {
                    local_cholfact[cholfact_index] = cholfact[cholfact_index * ndata + i];
                    cholfact_index++;
                }
            }
            double local_meas[mfeat], local_meas_unc[mfeat];
            for (int j = 0; j < mfeat; ++j) {
                local_meas[j] = meas[j * ndata + i];
                local_meas_unc[j] = meas_unc[j * ndata + i];
            }
            
            // propose a new value of chi
            Propose(local_chi, local_cholfact, proposed_chi, snorm_deviate, scaled_proposal, pchi);
            
            // get value of log-posterior for proposed chi value
            double logdens_meas_prop = LogDensity(proposed_chi, local_meas, local_meas_unc);
            double logdens_pop_prop = p_Theta->LogDensity(proposed_chi);
            double logpost_prop = logdens_meas_prop + logdens_pop_prop;
            
            /*
             * Uncomment this bit of code for help when debugging.
             *
             if (idata == 0) {
             printf("current iter, idata, mfeat, pchi, dtheta: %i, %i, %i, %i, %i\n", current_iter, idata, mfeat, pchi, dtheta);
             printf("  measurements: %g, %g, %g, %g, %g\n", local_meas[0], local_meas[1], local_meas[2], local_meas[3], local_meas[4]);
             printf("  measurement sigmas: %g, %g, %g, %g, %g\n", local_meas_unc[0], local_meas_unc[1], local_meas_unc[2],
             local_meas_unc[3], local_meas_unc[4]);
             printf("  cholfact: %g, %g, %g, %g, %g, %g\n", local_cholfact[0], local_cholfact[1], local_cholfact[2], local_cholfact[3],
             local_cholfact[4], local_cholfact[5]);
             printf("  current chi: %g, %g, %g\n", local_chi[0], local_chi[1], local_chi[2]);
             printf("  proposed chi: %g, %g, %g\n", proposed_chi[0], proposed_chi[1], proposed_chi[2]);
             printf("  current logdens_meas, logdens_pop: %g, %g\n", logdens_meas[idata], logdens_pop[idata]);
             printf("  proposed logdens_meas, logdens_pop: %g, %g\n", logdens_meas_prop, logdens_pop_prop);
             printf("\n");
             }
             */
            
            // accept the proposed value of the characteristic?
            double logdens_meas_i = logdens[i];
            double logdens_pop_i = logdens_pop[i];
            double logpost_current = logdens_meas_i + logdens_pop_i;
            double metro_ratio = 0.0;
            bool accept = AcceptProp(logpost_prop, logpost_current, 0.0, 0.0, metro_ratio);
            
            // adapt the covariance matrix of the characteristic proposal distribution
            AdaptProp(local_cholfact, snorm_deviate, scaled_proposal, metro_ratio, pchi, current_iter);
            
            for (int j = 0; j < dim_cholfact; ++j) {
                // copy value of this adapted cholesky factor back to global memory
                cholfact[j * ndata + i] = local_cholfact[j];
            }
            
            // copy local RNG state back to global memory
            devStates[idata] = localState;
            // printf("current iter, Accept, idata: %d, %d, %d\n", current_iter, accept, idata);
            for (int j=0; j<pchi; j++) {
                chi[ndata * j + i] = accept ? proposed_chi[j] : local_chi[j];
            }
            logdens[i] = accept ? logdens_meas_prop : logdens_meas_i;
            logdens_pop[i] = accept ? logdens_pop_prop : logdens_pop_i;
            naccept[i] += accept;
        }
        
        p_Theta->SetLogDens(logdens_pop);
        
	    current_iter++;
	}
    
    void Propose(double* p_chi, double* p_cholfact, double* p_proposed_chi, double* p_snorm_deviate,
                 double* p_scaled_proposal, int pchi)
    {
        // get the unit proposal
        for (int j=0; j<pchi; j++) {
            p_snorm_deviate[j] = snorm(rng);
        }
        
        // propose a new chi value
        int cholfact_index = 0;
        for (int j=0; j<pchi; j++) {
            double scaled_proposal_j = 0.0;
            for (int k=0; k<(j+1); k++) {
                // transform the unit proposal to the centered proposal, drawn from a multivariate normal.
                scaled_proposal_j += p_cholfact[cholfact_index] * p_snorm_deviate[k];
                cholfact_index++;
            }
            p_proposed_chi[j] = p_chi[j] + scaled_proposal_j;
            p_scaled_proposal[j] = scaled_proposal_j;
        }
    }
    
    bool AcceptProp(double logdens_prop, double logdens_current, double forward_dens,
                    double backward_dens, double& ratio)
    {
        double lograt = logdens_prop - forward_dens - (logdens_current - backward_dens);
        lograt = min(lograt, 0.0);
        ratio = exp(lograt);
        double unif = uniform(rng);
        bool accept = (unif < ratio) && isfinite(ratio);
        return accept;
    }

    void AdaptProp(double* p_cholfact, double* p_snorm_deviate, double* p_scaled_proposal, double metro_ratio,
                   int pchi, int current_iter)
    {
        double unit_norm = 0.0;
        for (int j=0; j<pchi; j++) {
            unit_norm += p_snorm_deviate[j] * p_snorm_deviate[j];
        }
        unit_norm = sqrt(unit_norm);
        double decay_sequence = 1.0 / pow(current_iter, decay_rate);
        double scaled_coef = sqrt(decay_sequence * fabs(metro_ratio - target_rate)) / unit_norm;
        for (int j=0; j<pchi; j++) {
            p_scaled_proposal[j] *= scaled_coef;
        }
        bool downdate = (metro_ratio < target_rate);
        // do rank-1 cholesky update to update the proposal covariance matrix
        chol_update_r1(p_cholfact, p_scaled_proposal, pchi, downdate);
    }
    
	void ResetAcceptance() {
		thrust::fill(d_naccept.begin(), d_naccept.end(), 0);
	}
    
	// setters and getters
	void SetChi(svector& new_chi, bool update_logdens = true) {
		chi = new_chi;
		if (update_logdens) {
            svector logdens_pop = p_Theta->GetLogDens();
            for (int i=0; i<ndata; i++) {
                double local_meas[mfeat], local_meas_unc[mfeat];
                for (int j = 0; j < mfeat; ++j) {
                    local_meas[j] = meas[j * ndata + i];
                    local_meas_unc[j] = meas_unc[j * ndata + i];
                }
                double local_chi[pchi];
                for (int j = 0; j<pchi; j++) {
                    local_chi[j] = chi[j * ndata + i];
                }
                logdens[i] = LogDensity(local_chi, local_meas, local_meas_unc);
                // now update the posteriors of the characteristics | population parameter
                logdens_pop[i] = p_Theta->LogDensity(local_chi);
            }
		}
	}
    
	// make sure that the data augmentation knows about the population parameters
	void SetPopulationPtr(boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > t) { p_Theta = t; }
    
	void SetLogDens(svector& new_logdens) {
		logdens = new_logdens;
	}
	void SetCholFact(svector& new_cholfact) {
		cholfact = new_cholfact;
	}
    
	void SetSaveTrace(bool dosave) { save_trace = dosave; }
    
	bool SaveTrace() { return save_trace; }
    
	// return the value of the characteristic in a std::vector of std::vectors for convenience
	vecvec GetChi() {
		vecvec new_chi(ndata);
		for (int i = 0; i < ndata; ++i) {
			// organize values into a 2-d array of dimensions ndata x pchi
			svector chi_i(pchi);
			for (int j = 0; j < pchi; ++j) {
				chi_i[j] = chi[ndata * j + i];
			}
			new_chi[i] = chi_i;
		}
		return new_chi;
	}
    
	svector GetLogDensVec() {return logdens;}
	double GetLogDens() {return std::accumulate(logdens.begin(), logdens.end(), 0.0);}
	hvector GetChiVec() {return chi;}
	int GetDataDim() {return ndata;}
	int GetChiDim() {return pchi;}
	boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > GetPopulationPtr() { return p_Theta; }
	thrust::host_vector<int> GetNaccept() {return naccept;}
    
protected:
	// measurements and their uncertainties
	svector meas;
	svector meas_unc;
	int ndata;
	// characteristics
	svector chi;
	// population-level parameters
	boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > p_Theta;
	// logarithm of conditional posterior densities, y | chi
	svector logdens;
	// cholesky factors of Metropolis proposal covariance matrix
	svector cholfact;
	// MCMC sampler parameters
	int current_iter;
    std::vector<int> naccept;
	bool save_trace;
};

#endif
