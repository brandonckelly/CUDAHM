/*
 * input_output.cpp
 *
 *  Created on: July 22, 2014
 *      Author: brandonkelly
 *      Editor: szalaigj (based on input_out.hpp of DustHM project)
 */

#include "input_output.hpp"
#include <cmath>
#include <stdexcept>

class BaseDataAdapter : public IDataAdapter
{
public:
	BaseDataAdapter()
		: m_thetasFileHeader(std::string()), m_chisFileHeader(std::string())
	{
	}

	BaseDataAdapter(std::string thetasFileHeader, std::string chisFileHeader)
		: m_thetasFileHeader(thetasFileHeader), m_chisFileHeader(chisFileHeader)
	{
	}

	// return the number of lines in a text file
	virtual int get_file_lines(std::string& filename) 
	{
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
	virtual void read_data(std::string& filename, vecvec& meas, vecvec& meas_unc, int ndata, int mfeat, bool hasHeader)
	{
		std::ifstream input_file(filename.c_str());
		meas.resize(ndata);
		meas_unc.resize(ndata);
		if (hasHeader) {
			std::string header_line;
			std::getline(input_file, header_line);  // read the header line
		}
		for (int i = 0; i < ndata; ++i) {
			std::vector<double> this_meas(mfeat);
			std::vector<double> this_meas_unc(mfeat);
			for (int j = 0; j < mfeat; ++j) {
				input_file >> this_meas[j] >> this_meas_unc[j];
			}
			meas[i] = this_meas;
			meas_unc[i] = this_meas_unc;
		}
		input_file.close();
	}

	// dump the sampled values of the population parameter to a text file
	virtual void write_thetas(std::string& filename, vecvec& theta_samples)
	{
		std::ofstream outfile(filename.c_str());
		if (!m_thetasFileHeader.empty()) {
			outfile << m_thetasFileHeader << std::endl;
		}
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
	virtual void write_chis(std::string& filename, std::vector<vecvec>& chi_samples)
	{
		std::ofstream outfile(filename.c_str());
		int nsamples = chi_samples.size();
		int ndata = chi_samples[0].size();
		int pchi = chi_samples[0][0].size();
		if (!m_chisFileHeader.empty()) {
			outfile << m_chisFileHeader << std::endl;
		}
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

private:
	std::string m_thetasFileHeader;
	std::string m_chisFileHeader;

};

class CBTDataAdapter : public BaseDataAdapter, public ICBTPlugin
{
public:
	CBTDataAdapter() : BaseDataAdapter()
	{
	}

	CBTDataAdapter(std::string thetasFileHeader, std::string chisFileHeader)
		: BaseDataAdapter(thetasFileHeader, chisFileHeader)
	{
	}

	// load in a set of (const, beta, temp) values
	virtual void load_cbt(std::string& filename, vecvec& cbt, int ndata)
	{
		std::ifstream input_file(filename.c_str());
		cbt.resize(ndata);
		for (int i = 0; i < ndata; ++i) {
			std::vector<double> this_cbt(3);
			for (int j = 0; j < 3; ++j) {
				input_file >> this_cbt[j];
			}
			cbt[i] = this_cbt;
		}
		input_file.close();
	}
};

class DistDataAdapter : public BaseDataAdapter, public IDistPlugin
{
public:
	DistDataAdapter() : BaseDataAdapter()
	{
	}

	DistDataAdapter(std::string thetasFileHeader, std::string chisFileHeader)
		: BaseDataAdapter(thetasFileHeader, chisFileHeader)
	{
	}

	// load in a set of distance values
	virtual void load_dist_data(std::string& filename, std::vector<double>& dists, int ndata)
	{
		std::ifstream input_file(filename.c_str());
		dists.resize(ndata);
		for (int i = 0; i < ndata; ++i) {
			input_file >> dists[i];
		}
		input_file.close();
	}
};