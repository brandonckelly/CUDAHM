//
//  main.cpp
//  dusthm-cpp
//
//  Created by Brandon Kelly on 4/30/14.
//  Copyright (c) 2014 Brandon C. Kelly. All rights reserved.
//

// standard library includes
#include <iostream>
#include <time.h>

// local CUDAHM includes
#include "GibbsSampler.hpp"
#include "input_output.hpp"
#include "ConstBetaTemp.hpp"
#include "DustPopPar.hpp"

const int mfeat = 5;
const int pchi = 3;  // chi = {log C, beta, log T}, where C \propto N_H (column density)
const int dtheta = 9;


int main(int argc, char** argv)
{
	time_t timer1, timer2;  // keep track of how long the program takes to run
	time(&timer1);
	/*
	 * Read in the data for the measurements, meas, and their standard deviations, meas_unc.
	 */
	std::string datafile = "/Users/brandonkelly/Projects/CUDAHM/dusthm/data/cbt_sed_1000.dat";
	int ndata = get_file_lines(datafile) - 1;  // subtract off one line for the header
	std::cout << "Loaded " << ndata << " data points." << std::endl;
    
	vecvec fnu(ndata);  // the measured SEDs
	vecvec fnu_sig(ndata);  // the standard deviation in the measurement errors
	read_data(datafile, fnu, fnu_sig, ndata, mfeat);
    
	/*
	 * Set the number of MCMC iterations and the amount of thinning for the chi and theta samples.
	 *
	 * NOTE THAT IF YOU HAVE A LARGE DATA SET, YOU WILL PROBABLY WANT TO THIN THE CHI VALUES SIGNIFICANTLY SO
	 * YOU DO NOT RUN OUR OF MEMORY.
	 */
    
	int nmcmc_iter = 50000;
	int nburnin = nmcmc_iter / 2;
	int nchi_samples = 100;
	int nthin_chi = nmcmc_iter / nchi_samples;
    
	// first create pointers to instantiated subclassed DataAugmentation and PopulationPar objects, since we need to give them to the
	// constructor for the GibbsSampler class.
	boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > CBT(new ConstBetaTemp<mfeat, pchi, dtheta>(fnu, fnu_sig));
	boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > Theta(new DustPopPar<mfeat, pchi, dtheta>);
    
	// instantiate the GibbsSampler object and run the sampler
	GibbsSampler<mfeat, pchi, dtheta> Sampler(CBT, Theta, nmcmc_iter, nburnin, nthin_chi);
    
	/*
	 * DEBUGGING
	 */
    //	vecvec cbt_true(ndata);
    //	std::string cbtfile = "../data/true_cbt_1000.dat";
    //	load_cbt(cbtfile, cbt_true, ndata);
    //
    //	// copy input data to data members
    //	hvector h_cbt(ndata * pchi);
    //	dvector d_cbt;
    //	for (int j = 0; j < pchi; ++j) {
    //		for (int i = 0; i < ndata; ++i) {
    //			h_cbt[ndata * j + i] = cbt_true[i][j];
    //		}
    //	}
    //	// copy data from host to device
    //	d_cbt = h_cbt;
    //
    //	hvector h_theta(dtheta);
    //	h_theta[0] = 15.0;
    //	h_theta[1] = 2.0;
    //	h_theta[2] = log(15.0);
    //	h_theta[3] = log(1.0);
    //	h_theta[4] = log(0.1);
    //	h_theta[5] = log(0.3);
    //	h_theta[6] = atanh(-0.5);
    //	h_theta[7] = atanh(0.0);
    //	h_theta[8] = atanh(0.25);
    //
    //	Sampler.GetDaugPtr()->SetChi(d_cbt, true);
    //	Sampler.GetThetaPtr()->SetTheta(h_theta, true);
    //	Sampler.FixPopPar();
    
	// run the MCMC sampler
	Sampler.Run();
    
    // grab the samples
	vecvec theta_samples = Sampler.GetPopSamples();
	std::vector<vecvec> chi_samples = Sampler.GetCharSamples();
    
	std::cout << "Writing results to text files..." << std::endl;
    
	// write the sampled theta values to a file. Output will have nsamples rows and dtheta columns.
	std::string thetafile("dusthm-cpp_thetas.dat");
	write_thetas(thetafile, theta_samples);
    
	// write the posterior means and standard deviations of the characteristics to a file. output will have ndata rows and
	// 2 * pchi columns, where the column format is posterior mean 1, posterior sigma 1, posterior mean 2, posterior sigma 2, etc.
	std::string chifile("dusthm-cpp_chi_summary.dat");
	write_chis(chifile, chi_samples);
    
	time(&timer2);
	double seconds = difftime(timer2, timer1);
    
	std::cout << "Program took " << seconds << " seconds." << std::endl;
    
}

