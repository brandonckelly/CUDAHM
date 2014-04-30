//
//  PopulationPar.hpp
//  dusthm-cpp
//
//  Created by Brandon Kelly on 4/30/14.
//  Copyright (c) 2014 Brandon C. Kelly. All rights reserved.
//

#ifndef dusthm_cpp_PopulationPar_hpp
#define dusthm_cpp_PopulationPar_hpp

#include "DataAugmentation.hpp"

// class for a population level parameter
template <int mfeat, int pchi, int dtheta>
class PopulationPar
{
public:
	// constructors
	PopulationPar() {
		theta.resize(dtheta);
		snorm_deviate.resize(dtheta);
		scaled_proposal.resize(dtheta);
		const int dim_cholfact = dtheta * dtheta - ((dtheta - 1) * dtheta) / 2;
		cholfact.resize(dim_cholfact);
		current_logdens = -1e300;
		current_iter = 0;
		naccept = 0;
	}
    
    virtual void LogDensity(double* chi, svector t) {return 0.0;}
    
	virtual void InitialValue() {
		// set initial value of theta to zero
		thrust::fill(theta.begin(), theta.end(), 0.0);
	}
    
	virtual void InitialCholFactor() {
		// set initial covariance matrix of the theta proposals as the identity matrix
		thrust::fill(cholfact.begin(), cholfact.end(), 0.0);
		int diag_index = 0;
		for (int k=0; k<dtheta; k++) {
			cholfact[diag_index] = 1.0;
			diag_index += k + 2;
		}
	}
    
	// calculate the initial value of the population parameters
	void Initialize() {
		InitialValue();
		InitialCholFactor();
        chi = p_Daug->GetChi()
        double local_chi[pchi];
        for (int i=0; i<ndata; i++) {
            for (int j=0; j<pchi; j++) {
                local_chi[j] = chi[i][j];
            }
            logdens[i] = LogDensity(local_chi, theta);
        }
        current_logdens = std::accumulate(logdens.begin(), logdens.end());
        
		// reset the number of MCMC iterations
		current_iter = 1;
		naccept = 0;
	}
    
	// return the log-prior of the population parameters
	virtual double LogPrior(svector t) { return 0.0; }
    
	// propose a new value of the population parameters
	virtual svector Propose() {
	    // get the unit proposal
	    for (int k=0; k<dtheta; k++) {
	        snorm_deviate[k] = snorm(rng);
	    }
        
	    // transform unit proposal so that is has a multivariate normal distribution
	    svector proposed_theta(dtheta);
        proposed_theta.fill(proposed_theta.begin(), proposed_theta.end(), 0.0);
	    int cholfact_index = 0;
	    for (int j=0; j<dtheta; j++) {
	        for (int k=0; k<(j+1); k++) {
	        	// cholfact is lower-diagonal matrix stored as a 1-d array
	            scaled_proposal[j] += cholfact[cholfact_index] * snorm_deviate[k];
	            cholfact_index++;
	        }
	        proposed_theta[j] = theta[j] + scaled_proposal[j];
	    }
        
	    return proposed_theta;
	}
    
	// adapt the covariance matrix (i.e., the cholesky factors) of the theta proposals
	virtual void AdaptProp(double metro_ratio) {
		double unit_norm = 0.0;
	    for (int j=0; j<dtheta; j++) {
	    	unit_norm += snorm_deviate[j] * snorm_deviate[j];
	    }
	    unit_norm = sqrt(unit_norm);
	    double decay_sequence = 1.0 / std::pow(current_iter, decay_rate);
	    double scaled_coef = sqrt(decay_sequence * fabs(metro_ratio - target_rate)) / unit_norm;
	    for (int j=0; j<dtheta; j++) {
	        scaled_proposal[j] *= scaled_coef;
	    }
        
	    bool downdate = (metro_ratio < target_rate);
	    double* p_cholfact = cholfact.data();
        double* p_scaled_proposal = scaled_proposal.data();
	    // rank-1 update of the cholesky factor
	    chol_update_r1(p_cholfact, p_scaled_proposal, dtheta, downdate);
	}
    
	// calculate whether to accept or reject the metropolist-hastings proposal
	bool AcceptProp(double logdens_prop, double logdens_current, double& ratio, double forward_dens = 0.0,
                    double backward_dens = 0.0) {
	    double lograt = logdens_prop - forward_dens - (logdens_current - backward_dens);
	    lograt = std::min(lograt, 0.0);
	    ratio = exp(lograt);
	    double unif = uniform(rng);
	    bool accept = (unif < ratio) && isfinite(ratio);
	    if (!isfinite(ratio)) {
			// metropolis ratio is not finite, so make it equal to zero to not screw up the proposal cholesky factor update
	    	ratio = 0.0;
		}
	    return accept;
	}
    
	// update the value of the population parameter value using a robust adaptive metropolis algorithm
	virtual void Update() {
		// get current conditional log-posterior of population
		double logdens_current = std::accumulate(logdens.begin(), logdens.end(), 0.0);
		logdens_current += LogPrior(theta);
        
		// propose new value of population parameter
		svector proposed_theta = Propose();
        
        svector chi = p_Daug->GetChiVec();
        
		// calculate log-posterior of new population parameter
        double p_chi[pchi]
        for (int i=0; i<ndata; i++) {
            for (int j=0; j<pchi; j++) {
                p_chi[j] = chi[j * ndata + i]
            }
            logdens_prop[i] = LogDensity(p_chi, proposed_theta)
        }
		double logdens_prop = std::accumulate(proposed_logdens.begin(), proposed_logdens.end());
        
		logdens_prop += LogPrior(proposed_theta);
        
		// accept the proposed value?
		double metro_ratio = 0.0;
		bool accept = AcceptProp(logdens_prop, logdens_current, metro_ratio);
		if (accept) {
			theta = proposed_theta;
            logdens = proposed_logdens;
			naccept++;
			current_logdens = logdens_prop;
		} else {
			current_logdens = logdens_current;
		}
        
		// adapt the covariance matrix of the proposals
		AdaptProp(metro_ratio);
		current_iter++;
	}
    
	void ResetAcceptance() { naccept = 0; }
    
	// setters and getters
	void SetDataAugPtr(boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > DataAug) {
		p_Daug = DataAug;
		ndata = Daug->GetDataDim();
		logdens.resize(ndata);
		proposed_logdens.resize(ndata);
	}
    
	void SetTheta(svector& t, bool update_logdens = true) {
		theta = t;
		if (update_logdens) {
			// update value of conditional log-posterior for theta|chi
			svector chi = Daug->GetChiVec();
            // calculate log-posterior of new population parameter
            double p_chi[pchi]
            for (int i=0; i<ndata; i++) {
                for (int j=0; j<pchi; j++) {
                    p_chi[j] = chi[j * ndata + i]
                }
                logdens[i] = LogDensity(p_chi, theta)
            }
            double logdens_prop = std::accumulate(logdens.begin(), logdens.end());
            logdens_prop += LogPrior(theta);
		}
	}
    
	void SetLogDens(svector& new_logdens) {
		logdens = new_logdens;
		current_logdens = std::accumulate(logdens.begin(), logdens.end());
	}
	void SetCholFact(svector cholfact_new) { cholfact = cholfact_new; }
	void SetCurrentIter(int iter) { current_iter = iter; }
    
	svector GetTheta() { return theta; }
	double GetLogDens() { return current_logdens; } // return the current value of summed log p(chi | theta);
	svector GetLogDensVec() {return logdens;}
	int GetDim() { return dtheta; }
	svector GetCholFactor() { return cholfact; }
	int GetNaccept() { return naccept; }
	boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > GetDataAugPtr() { return p_Daug; }
    
protected:
	// the value of the population parameter
	svector theta;
	// log of the value the probability of the characteristics given the population parameter, chi | theta
	svector logdens;
	svector proposed_logdens;
	double current_logdens;
	// make sure that the population parameter knows about the characteristics
	boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > p_Daug;
	// cholesky factors of Metropolis proposal covariance matrix
	svector cholfact;
	// interval variables used in robust adaptive metropolis algorithm
	svector snorm_deviate;
	svector scaled_proposal;
	// MCMC parameters
	int naccept;
	int current_iter;
	int ndata;
};

#endif
