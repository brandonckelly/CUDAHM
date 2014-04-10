/*
 * input_output.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: brandonkelly
 */

#ifndef INPUT_OUTPUT_HPP_
#define INPUT_OUTPUT_HPP_

// standard includes
#include <fstream>
#include <string>
#include <vector>

typedef std::vector<std::vector<double> > vecvec;


// return the number of lines in a text file
int get_file_lines(std::string& filename);

// read in the data
void read_data(std::string& filename, vecvec& meas, vecvec& meas_unc, int ndata, int mfeat);

// load in a set of (const, beta, temp) values
void load_cbt(std::string& filename, vecvec& cbt, int ndata);

// dump the sampled values of the population parameter to a text file
void write_thetas(std::string& filename, vecvec& theta_samples);

// dump the posterior means and standard deviations of the characteristics to a text file
void write_chis(std::string& filename, std::vector<vecvec>& chi_samples);

#endif /* INPUT_OUTPUT_HPP_ */
