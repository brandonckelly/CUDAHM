# executing e.g. python plot_chis_funcs.py b-1.2_l1.0_u100.0_fluxes_cnt_100000.dat b-1.2_l1.0_u100.0_filtered_fluxes_w_G_noise_mu_0.0_sig_1e-10_cnt_100000.dat b-1.2_l1.0_u100.0_init_b-1.901_l24.009_u69.99_lumfunc_chi_summary.dat b-1.2_l1.0_u100.0_init_b-1.901_l24.009_u69.99_
import argparse as argp
import numpy as np
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("real_flux_file", help="The file name of real flux data file.", type = str)
parser.add_argument("noisy_flux_file", help="The file name of noisy flux data file.", type = str)
parser.add_argument("estimated_flux_file", help="The file name of estimated flux data file.", type = str)
parser.add_argument("prefix", help="The prefic for created output files.", type = str)

args = parser.parse_args()
real_flux_file = args.real_flux_file
noisy_flux_file = args.noisy_flux_file
estimated_flux_file = args.estimated_flux_file
prefix = args.prefix

real_flux_data=np.loadtxt(real_flux_file)
noisy_flux_data=np.loadtxt(noisy_flux_file,delimiter=' ',usecols=(0,1))
noisy_flux_data_0=noisy_flux_data[:,0]
estimated_flux_data=np.loadtxt(estimated_flux_file,delimiter=' ',usecols=(0,1))
estimated_flux_data_0=estimated_flux_data[:,0]

print real_flux_data.min()
print real_flux_data.max()
#print noisy_flux_data_0[0]
#print estimated_flux_data_0[0]

lbl_real = 'real fluxes'
lbl_noisy = 'noisy fluxes'
lbl_estimated = 'estimated fluxes'
lbl_noisy_estimated = 'noisy-estimated fluxes'

#fig = figure()	
#ax = fig.add_subplot(1,1,1, xlim=[real_flux_data.min(), real_flux_data.max()], ylim=[noisy_flux_data_0.min(), noisy_flux_data_0.max()]) # one row, one column, first plot
#ax.scatter(real_flux_data,noisy_flux_data_0, marker = ".", linewidth=0.01)
#ax.set_xlabel(lbl_real)
#ax.set_ylabel(lbl_noisy)
#savefig(prefix + 'real_noisy.png',dpi=120)
#
#fig = figure()
#ax = fig.add_subplot(1,1,1, xlim=[real_flux_data.min(), real_flux_data.max()], ylim=[estimated_flux_data_0.min(), estimated_flux_data_0.max()]) # one row, one column, first plot
#ax.scatter(real_flux_data,estimated_flux_data_0, color = 'g', marker = ".", linewidth=0.01)
#ax.set_xlabel(lbl_real)
#ax.set_ylabel(lbl_estimated)
#savefig(prefix + 'real_estimated.png',dpi=120)

fig = figure()
#ax = fig.add_subplot(1,1,1, xlim=[real_flux_data.min(), real_flux_data.max()], ylim=[noisy_flux_data_0.min(), noisy_flux_data_0.max()]) # one row, one column, first plot
#ax.scatter(real_flux_data,noisy_flux_data_0, label = 'real-noisy', color = 'b', marker = ".", linewidth=0.01)
ax2 = fig.add_subplot(1,1,1, xlim=[real_flux_data.min(), real_flux_data.max()], ylim=[estimated_flux_data_0.min(), estimated_flux_data_0.max()]) # one row, one column, first plot
ax2.scatter(real_flux_data,noisy_flux_data_0, label = 'real-noisy', color = 'b', marker = ".", linewidth=0.01)
ax2.scatter(real_flux_data,estimated_flux_data_0, label = 'real-estimated', color = 'g', marker = ".", linewidth=0.01)
ax2.set_xlabel(lbl_real)
ax2.set_ylabel(lbl_noisy_estimated)
ax2.legend(loc=1)
savefig(prefix + 'real_noisy_estimated.png',dpi=120)