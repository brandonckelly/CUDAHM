# executing e.g. python _comp_mse_for_max_like_and_post_mean.py fluxes_cnt_100000.dat filtered_fluxes_w_G_noise_mu_0.0_sig_1.0_fl_5.0_cnt_100000.dat lumfunc_chi_summary.dat

import argparse as argp
import datetime as dt
import numpy as np

parser = argp.ArgumentParser()
parser.add_argument("real_flux_file", help="The file name of real flux data file.", type = str)
parser.add_argument("noisy_flux_file", help="The file name of noisy flux data file.", type = str)
parser.add_argument("estimated_flux_file", help="The file name of estimated flux data file.", type = str)
parser.add_argument("--n_regimes", default = 10, help="The number of flux regimes.", type = int)

args = parser.parse_args()
real_flux_file = args.real_flux_file
noisy_flux_file = args.noisy_flux_file
estimated_flux_file = args.estimated_flux_file
n_regimes = args.n_regimes

# Wider margins to allow for larger labels; may need to adjust left:
rc('figure.subplot', bottom=.125, top=.95, right=.95)  # left=0.125

# Optionally make default line width thicker:
#rc('lines', linewidth=2.0) # doesn't affect frame lines

rc('font', size=14)  # default for labels (not axis labels)
rc('font', family='serif')  # default for labels (not axis labels)
rc('axes', labelsize=18)
rc('xtick.major', pad=8)
rc('xtick', labelsize=14)
rc('ytick.major', pad=8)
rc('ytick', labelsize=14)

rc('savefig', dpi=150)  # mpl's default dpi is 100
rc('axes.formatter', limits=(-4,4))

# Use TeX labels with CMR font:
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

real_flux_data=np.loadtxt(real_flux_file)
noisy_flux_data=np.loadtxt(noisy_flux_file,delimiter=' ',usecols=(0,1))
noisy_flux_data_0=noisy_flux_data[:,0]
estimated_flux_data=np.loadtxt(estimated_flux_file,delimiter=' ',usecols=(0,1))
estimated_flux_data_0=estimated_flux_data[:,0]

fluxes_list=[]
for idx in range(0, len(real_flux_data)):
  fluxes_list.append((real_flux_data[idx],noisy_flux_data_0[idx],estimated_flux_data_0[idx]))

fluxes = np.array(fluxes_list)
sorted_fluxes = np.sort(fluxes,axis=0)

rate=len(sorted_fluxes)/n_regimes

mse_max_like_list = []
mse_post_mean_list = []

for idx in range(1,n_regimes+1):
  lim_reg_lower = rate * (idx - 1)
  lim_reg_upper = rate * idx
  current_mse_max_like = 0.0
  current_mse_post_mean = 0.0
  for subIdx in range(lim_reg_lower,lim_reg_upper):
    current_mse_max_like += (sorted_fluxes[subIdx][1] - sorted_fluxes[subIdx][0])**2.0
    current_mse_post_mean += (sorted_fluxes[subIdx][2] - sorted_fluxes[subIdx][0])**2.0
  current_mse_max_like /= rate
  current_mse_post_mean /= rate
  mse_max_like_list.append(current_mse_max_like)
  mse_post_mean_list.append(current_mse_post_mean)
  
print "MSE of max like - true", mse_max_like_list
print "MSE of post mean - true", mse_post_mean_list