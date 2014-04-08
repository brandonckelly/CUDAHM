/*
 * input_output.cpp
 *
 *  Created on: Mar 21, 2014
 *      Author: brandonkelly
 */

#include "input_output.hpp"
#include <cmath>
#include <stdexcept>

// return the number of lines in a text file
int get_file_lines(std::string& filename) {
    int number_of_lines = 0;
    std::string line;
    std::ifstream inputfile(filename.c_str());

    if (inputfile.good()) {
        while (std::getline(inputfile, line))
            ++number_of_lines;
        inputfile.close();
        return number_of_lines;
	} else {
		std::string errmsg("File ");
		errmsg.append(filename);
		errmsg.append(" does not exist.\n");
		throw std::runtime_error(errmsg);
	}
}

// read in the data
void read_data(std::string& filename, vecvec& meas, vecvec& meas_unc, int ndata, int mfeat) {
	std::ifstream input_file(filename.c_str());
	meas.resize(ndata);
	meas_unc.resize(ndata);
	for (int i = 0; i < ndata; ++i) {
		std::vector<double> this_meas(mfeat);
		std::vector<double> this_meas_unc(mfeat);
		for (int j = 0; j < mfeat; ++j) {
			input_file >> this_meas[j] >> this_meas_unc[j];
		}
		meas[i] = this_meas;
		meas_unc[i] = this_meas;
	}
	input_file.close();
}

// dump the sampled values of the population parameter to a text file
void write_thetas(std::string& filename, vecvec& theta_samples) {
	std::ofstream outfile(filename.c_str());

	outfile << "# log(C) mean, beta mean, log(T) mean, log(log(C) sigma), log(beta sigma), log(log(T) sigma), tanh(log(C) corr), "
			<< "tanh(beta corr), tanh(log(T) corr)" << std::endl;

	int nsamples = theta_samples.size();
	int dtheta = theta_samples[0].size();
	for (int i = 0; i < nsamples; ++i) {
		for (int j = 0; j < dtheta; ++j) {
			outfile << theta_samples[i][j] << " ";
		}
		outfile << std::endl;
	}
}

// dump the posterior means and standard deviations of the characteristics to a text file
void write_chis(std::string& filename, std::vector<vecvec>& chi_samples) {
	std::ofstream outfile(filename.c_str());
	int nsamples = chi_samples.size();
	int ndata = chi_samples[0].size();
	int pchi = chi_samples[0][0].size();

	outfile << "# log(C_i) mean, log(C_i) sigma, beta_i mean, beta_i sigma, log(T_i) mean, log(T_i) sigma" << std::endl;

	for (int i = 0; i < ndata; ++i) {
		std::vector<double> post_mean_i(pchi, 0.0);
		std::vector<double> post_msqr_i(pchi, 0.0);  // posterior mean of the square of the values for chi_i
		for (int j = 0; j < nsamples; ++j) {
			for (int k = 0; k < pchi; ++k) {
				post_mean_i[k] += chi_samples[j][i][k] / nsamples;
				post_msqr_i[k] += chi_samples[j][i][k] * chi_samples[j][i][k] / nsamples;
			}
		}
		for (int k = 0; k < pchi; ++k) {
			double post_sigma_ik = sqrt(post_msqr_i[k] - post_mean_i[k] * post_mean_i[k]);  // posterior standard deviation
			outfile << post_mean_i[k] << " " << post_sigma_ik <<" ";
		}
		outfile << std::endl;
	}

	outfile.close();
}
