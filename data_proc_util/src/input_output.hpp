/*
 * input_output.hpp
 *
 *  Created on: July 22, 2014
 *      Author: brandonkelly
 *      Editor: szalaigj (based on input_out.hpp of DustHM project)
 */

#ifndef INPUT_OUTPUT_HPP_
#define INPUT_OUTPUT_HPP_

// standard includes
#include <fstream>
#include <string>
#include <vector>

typedef std::vector<std::vector<double> > vecvec;

// This is an interface for data loading, writing.
class IDataAdapter
{
public:
	// return the number of lines in a text file
	virtual int get_file_lines(std::string& filename) = 0;

	// read in the data
	virtual void read_data(std::string& filename, vecvec& meas, vecvec& meas_unc, int ndata, int mfeat, bool hasHeader) = 0;
	
	// dump the sampled values of the population parameter to a text file
	virtual void write_thetas(std::string& filename, vecvec& theta_samples) = 0;

	// dump the posterior means and standard deviations of the characteristics to a text file
	virtual void write_chis(std::string& filename, std::vector<vecvec>& chi_samples) = 0;
};

// This is an interface which is a plugin for IDataAdapter and it is extended by
// special load_cbt function which is used by DustHM project.
class ICBTPlugin
{
public:
	// load in a set of (const, beta, temp) values
	virtual void load_cbt(std::string& filename, vecvec& cbt, int ndata) = 0;
};

// This is an interface which is a plugin for IDataAdapter and it is extended by
// special load_dist_data function which is used by LumFuncHM project.
class IDistPlugin
{
public:
	// load in a set of distance values
	virtual void load_dist_data(std::string& filename, std::vector<double>& dists, int ndata) = 0;
};

#endif /* INPUT_OUTPUT_HPP_ */