# executing e.g. python plot_chis_funcs.py b-1.2_l1.0_u100.0_fluxes_cnt_100000.dat b-1.2_l1.0_u100.0_filtered_fluxes_w_G_noise_mu_0.0_sig_1e-10_cnt_100000.dat b-1.2_l1.0_u100.0_init_b-1.3_l5.0_u110.0_lumfunc_chi_summary.dat b-1.2_l1.0_u100.0_init_b-1.3_l5.0_u110.0_

# executing e.g. python plot_chis_funcs.py b-1.2_l1.0_u100.0_fluxes_cnt_100000.dat b-1.2_l1.0_u100.0_filtered_fluxes_w_G_noise_mu_0.0_sig_1e-10_cnt_100000.dat b-1.2_l1.0_u100.0_init_b-1.901_l24.009_u69.99_lumfunc_chi_summary.dat b-1.2_l1.0_u100.0_init_b-1.901_l24.009_u69.99_

# executing e.g. python plot_chis_funcs.py b-1.5_l1.0_u100.0_fluxes_cnt_100000.dat b-1.5_l1.0_u100.0_filtered_fluxes_w_G_noise_mu_0.0_sig_1e-10_cnt_100000.dat  b-1.5_l1.0_u100.0_init_b-1.25_l25.0_u75.0_lumfunc_chi_summary.dat b-1.5_l1.0_u100.0_init_b-1.25_l25.0_u75.0_

# executing e.g. python plot_chis_funcs.py b-1.5_l1.0_u100.0_fluxes_cnt_100000.dat b-1.5_l1.0_u100.0_filtered_fluxes_w_G_noise_mu_0.0_sig_1e-10_cnt_100000.dat  b-1.5_l1.0_u100.0_init_b-1.41_l2.0_u98.0_lumfunc_chi_summary.dat b-1.5_l1.0_u100.0_init_b-1.41_l2.0_u98.0_

# executing e.g. python plot_chis_funcs.py b-1.9_l25.0_u70.0_fluxes_cnt_100000.dat b-1.9_l25.0_u70.0_filtered_fluxes_w_G_noise_mu_0.0_sig_1e-10_cnt_100000.dat b-1.9_l25.0_u70.0_init_b-1.901_l24.009_u69.99_lumfunc_chi_summary.dat b-1.9_l25.0_u70.0_init_b-1.901_l24.009_u69.99_

import argparse as argp
import numpy as np
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("real_flux_file", help="The file name of real flux data file.", type = str)
parser.add_argument("noisy_flux_file", help="The file name of noisy flux data file.", type = str)
parser.add_argument("estimated_flux_file", help="The file name of estimated flux data file.", type = str)
parser.add_argument("prefix", help="The prefix for created output files.", type = str)

args = parser.parse_args()
real_flux_file = args.real_flux_file
noisy_flux_file = args.noisy_flux_file
estimated_flux_file = args.estimated_flux_file
prefix = args.prefix

# Use TeX labels with CMR font:
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

real_flux_data=np.loadtxt(real_flux_file)
noisy_flux_data=np.loadtxt(noisy_flux_file,delimiter=' ',usecols=(0,1))
noisy_flux_data_0=noisy_flux_data[:,0]
estimated_flux_data=np.loadtxt(estimated_flux_file,delimiter=' ',usecols=(0,1))
estimated_flux_data_0=estimated_flux_data[:,0]

print real_flux_data.min()
print real_flux_data.max()

lbl_real = 'log real fluxes'
lbl_noisy = 'log noisy fluxes'
lbl_estimated = 'log estimated fluxes'
lbl_noisy_estimated = 'log noisy-estimated fluxes'

mask = noisy_flux_data_0 > 0
filtered_noisy_flux_data_0 = noisy_flux_data_0[mask]
filtered_real_flux_data = real_flux_data[mask]

xlim_min = np.log(real_flux_data.min())
xlim_max = np.log(real_flux_data.max())
ylim_min = np.log(min(estimated_flux_data_0.min(), filtered_noisy_flux_data_0.min()))
ylim_max = np.log(max(estimated_flux_data_0.max(),filtered_noisy_flux_data_0.max()))

fig = figure()
ax = fig.add_subplot(1,1,1, xlim=[xlim_min, xlim_max], ylim=[ylim_min, ylim_max]) # one row, one column, first plot
ax.scatter(np.log(filtered_real_flux_data),np.log(filtered_noisy_flux_data_0), label = 'log real-noisy', color = 'b', marker = ".", linewidth=0.01, zorder=1)
ax.scatter(np.log(real_flux_data),np.log(estimated_flux_data_0), label = 'log real-estimated', color = 'g', marker = ".", linewidth=0.01, zorder=1)
ax.scatter(np.log(real_flux_data),np.log(real_flux_data), label = 'log real-real', color = 'r', marker = ".", linewidth=0.01, zorder=2)
ax.set_xlabel(lbl_real)
ax.set_ylabel(lbl_noisy_estimated)
ax.legend(loc=2)
ttl_array = prefix.split("_")
ttl = r'Real $\theta$: (' + ttl_array[0][1:] + ',' + ttl_array[1][1:] + ',' + ttl_array[2][1:] + ')' + r'\\Init. values of $\theta$: (' + ttl_array[4][1:] + ',' + ttl_array[5][1:] + ',' + ttl_array[6][1:] + ')'
suptitle(ttl)
savefig(prefix + 'log_real_noisy_estimated.png',dpi=120)