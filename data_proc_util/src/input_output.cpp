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
#include <iostream>

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

	void read_settings(std::string& settingsfilename, std::map<std::string, std::string>& props)
	{
		std::string delimiter = "=";
		std::ifstream input_file(settingsfilename.c_str());
		std::string line;
		if (input_file.good()) {
			while (std::getline(input_file, line))
			{
				int pos = line.find(delimiter);
				std::string propKey = line.substr(0, pos);
				std::string propValue = line.substr(pos + 1, line.length());
				props[propKey] = propValue;
			}
			input_file.close();
		}
		else {
			std::string errmsg("File ");
			errmsg.append(settingsfilename);
			errmsg.append(" does not exist.\n");
			throw std::runtime_error(errmsg);
		}
	}

	// return the number of lines in a text file
	int get_file_lines(std::string& filename) 
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
	void read_data(std::string& filename, vecvec& meas, vecvec& meas_unc, int ndata, int mfeat, bool hasHeader)
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
	void write_thetas(std::string& filename, const double * theta_samples, int nsamples, int dtheta)
	{
		std::ofstream outfile(filename.c_str());
		if (!m_thetasFileHeader.empty()) {
			outfile << m_thetasFileHeader << std::endl;
		}
		for (int i = 0; i < nsamples; ++i) {
			for (int j = 0; j < dtheta; ++j) {
				outfile << theta_samples[nsamples*j + i] << " ";
			}
			outfile << std::endl;
		}
	}

	// dump the posterior means and standard deviations of the characteristics to a text file
	void write_chis(std::string& filename, const double * chi_samples, int nsamples, int ndata, int pchi)
	{
		std::ofstream outfile(filename.c_str());
		if (!m_chisFileHeader.empty()) {
			outfile << m_chisFileHeader << std::endl;
		}
		for (int i = 0; i < ndata; ++i) {
			std::vector<double> post_mean_i(pchi, 0.0);
			std::vector<double> post_msqr_i(pchi, 0.0);  // posterior mean of the square of the values for chi_i
			for (int j = 0; j < nsamples; ++j) {
				for (int k = 0; k < pchi; ++k) {
					post_mean_i[k] += chi_samples[(k*ndata + i)*nsamples + j] / nsamples;
					post_msqr_i[k] += chi_samples[(k*ndata + i)*nsamples + j] * chi_samples[(k*ndata + i)*nsamples + j] / nsamples;
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
	void load_cbt(std::string& filename, vecvec& cbt, int ndata)
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
	void load_dist_data(std::string& filename, std::vector<double>& dists, int ndata)
	{
		std::ifstream input_file(filename.c_str());
		dists.resize(ndata);
		for (int i = 0; i < ndata; ++i) {
			input_file >> dists[i];
		}
		input_file.close();
	}

	// dump the traces of the lowest, the middle and the highest values of the characteristics to a text file
	// it is supposed there is only one characteristic for each object (so pchi == 1)
	void write_relevant_chis(std::string& filenameMin, std::string& filenameMax, std::string& filenameMedian, const double * chi_samples, int nsamples, int ndata)
	{
		std::ofstream outfileMin(filenameMin.c_str());
		std::ofstream outfileMax(filenameMax.c_str());
		std::ofstream outfileMedian(filenameMedian.c_str());
		int minPos, maxPos, medianPos;
		double minValue = 1e300;
		double maxValue = -1e300;
		double mean = 0.0;
		double msqr = 0.0;
		double chiActual;
		for (int i = 0; i < ndata; ++i)
		{
			// we find in the last sample
			chiActual = chi_samples[i*nsamples + nsamples - 1];
			if (chiActual < minValue)
			{
				minValue = chiActual;
				minPos = i;
			}
			if (chiActual > maxValue)
			{
				maxValue = chiActual;
				maxPos = i;
			}
			mean += chiActual / ndata;
			msqr += chiActual * chiActual / ndata;
		}
		double stDev = sqrt(msqr - mean * mean);
		double medianApprox;
		for (int i = 0; i < ndata; ++i)
		{
			// we find in the last sample
			chiActual = chi_samples[i*nsamples + nsamples - 1];
			if ((chiActual - mean > -stDev) && (chiActual - mean < stDev))
			{
				medianApprox = chiActual;
				medianPos = i;
				break;
			}
		}
		std::cout << minValue << " " << maxValue << " " << medianApprox << std::endl;
		for (int j = 0; j < nsamples; ++j) {
			outfileMin << chi_samples[minPos*nsamples + j] << std::endl;
			outfileMax << chi_samples[maxPos*nsamples + j] << std::endl;
			outfileMedian << chi_samples[medianPos*nsamples + j] << std::endl;
		}
		outfileMin.close();
		outfileMax.close();
		outfileMedian.close();
	}
};
