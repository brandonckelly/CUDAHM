//
//  ConstBetaTemp.hpp
//  dusthm-cpp
//
//  Created by Brandon Kelly on 4/30/14.
//  Copyright (c) 2014 Brandon C. Kelly. All rights reserved.
//

#ifndef dusthm_cpp_ConstBetaTemp_hpp
#define dusthm_cpp_ConstBetaTemp_hpp

#include "DataAugmentation.hpp"

// frequencies correspond to {500, 350, 250, 170, 70} microns, the Herschel bands
const double nu[5] = {5.99584916e11, 8.56549880e11, 1.19916983e12, 1.87370286e12, 2.99792458e12};

double modified_blackbody(double freq, double C, double beta, double T);

// Subclass the DataAugmentation class to override the method to generate the initial values of chi = (log C, beta, log T).
// Note that because the classes are templated, we need to use the 'this' pointer in order to access the data members from
// the base class.
template <int mfeat,int pchi,int dtheta>
class ConstBetaTemp: public DataAugmentation<mfeat, pchi, dtheta> {
public:
    
	ConstBetaTemp(vecvec& meas, vecvec& meas_unc) : DataAugmentation<mfeat, pchi, dtheta>(meas, meas_unc) {}
    
    double LogDensity(double* p_chi, double* p_meas, double* p_meas_unc)
    {
        double C = exp(p_chi[0]);
        double T = exp(p_chi[2]);
        double logdens_meas = 0.0;
        for (int j = 0; j < mfeat; ++j) {
            // p(y_ij | chi_ij) is a normal density centered at the model SED
            double model_sed = modified_blackbody(nu[j], C, p_chi[1], T);
            logdens_meas += -0.5 * (p_meas[j] - model_sed) * (p_meas[j] - model_sed) / (p_meas_unc[j] * p_meas_unc[j]);
        }
        return logdens_meas;
    }
    
	void Initialize() {
        for (int i=0; i<this->ndata; i++) {
            double cbt_mu[3] = {15.0, 2.0, log(15.0)};
            double cbt_sigma[3] = {4.6, 0.5, 0.33};
            for (int j = 0; j < pchi; ++j) {
                // randomly initialize chi = (log C, beta, log T) from a normal distribution
                this->chi[i + j * this->ndata] = cbt_mu[j] + cbt_sigma[j] * snorm(rng);
            }
            
            // set initial covariance matrix of the chi proposals as the identity matrix
            int diag_index = 0;
            for (int j=0; j<pchi; j++) {
                this->cholfact[i + this->ndata * diag_index] = 0.01;
                diag_index += j + 2;  // cholesky factor is lower diagonal
            }
            // copy value to registers before computing initial log-density
            double this_chi[pchi];
            for (int j = 0; j < pchi; ++j) {
                this_chi[j] = this->chi[j * this->ndata + i];
            }
            double local_meas[mfeat], local_meas_unc[mfeat];
            for (int j = 0; j < mfeat; ++j) {
                local_meas[j] = this->meas[j * this->ndata + i];
                local_meas_unc[j] = this->meas_unc[j * this->ndata + i];
            }
            this->logdens[i] = LogDensity(this_chi, local_meas, local_meas_unc);
        }
        std::fill(this->naccept.begin(), this->naccept.end(), 0);
		this->current_iter = 1;
	}
};

#endif
