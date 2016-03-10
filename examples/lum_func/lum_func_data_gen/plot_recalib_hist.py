# executing e.g.: python plot_recalib_hist.py fluxes_cnt_100000.dat filtered_fluxes_w_G_noise_mu_0.0_sig_1.0_fl_5.0_cnt_100000.dat lumfunc_chi_summary_2.dat --pdf_format False
import argparse as argp
import datetime as dt
import numpy as np
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("real_flux_file", help="The file name of real flux data file.", type = str)
parser.add_argument("noisy_flux_file", help="The file name of noisy flux data file.", type = str)
parser.add_argument("estimated_flux_file", help="The file name of estimated flux data file.", type = str)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
real_flux_file = args.real_flux_file
noisy_flux_file = args.noisy_flux_file
estimated_flux_file = args.estimated_flux_file
pdf_format = eval(args.pdf_format)

real_flux_data=np.loadtxt(real_flux_file)
noisy_flux_data=np.loadtxt(noisy_flux_file,delimiter=' ',usecols=(0,1))
noisy_flux_data_0=noisy_flux_data[:,0]
estimated_flux_data=np.loadtxt(estimated_flux_file,delimiter=' ',usecols=(0,1))
estimated_flux_data_0=estimated_flux_data[:,0]

fluxes_list=[]
for idx in range(0, len(real_flux_data)):
  fluxes_list.append((real_flux_data[idx],noisy_flux_data_0[idx],estimated_flux_data_0[idx]))

fluxes = np.array(fluxes_list)

arg_sort = np.argsort(fluxes, axis=0)

F_real = fluxes[arg_sort[:,0],0].flatten()
F_obs = fluxes[arg_sort[:,0],1].flatten()
F_recal = fluxes[arg_sort[:,0],2].flatten()

execfile("rc_settings.py")
rc('figure.subplot', bottom=.2, top=.92, right=.95, left=0.25)
rc('figure', figsize=(2.5, 2.5))
if(pdf_format!=True):
  rc('savefig', dpi=100)
  
fig, ax = subplots()

r = 1000
hist, bins = np.histogram(F_obs-F_real, bins=r)
ax.plot(bins[:-1], hist)
hist, bins = np.histogram(F_recal-F_real, bins=r)
ax.plot(bins[:-1], hist)
ax.set_xlim([-10,10])
ax.set_xlabel('$\Delta F$')
ax.set_ylabel('counts')

if(pdf_format):
  savefig('recalib_hist.pdf', format='pdf')
else:
  savefig('recalib_hist.png')