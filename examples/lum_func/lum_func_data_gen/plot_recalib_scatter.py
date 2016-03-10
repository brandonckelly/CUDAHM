# executing e.g.: python plot_recalib_scatter.py fluxes_cnt_100000.dat filtered_fluxes_w_G_noise_mu_0.0_sig_1.0_fl_5.0_cnt_100000.dat lumfunc_chi_summary_2.dat --pdf_format False
import argparse as argp
import datetime as dt
import numpy as np
from matplotlib.cbook import get_sample_data
from matplotlib.pyplot import *
import os

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

t0 = dt.datetime.today()

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

zz = np.arange(0, 5.1, 0.1)

limit = [0, 2.5]

execfile("rc_settings.py")
if(pdf_format!=True):
  rc('savefig', dpi=100)
rc('figure', figsize=(2.5, 2.5))
rc('figure.subplot', bottom=0.0, top=1.0, right=1.0, left=0.0)

fig, ax = subplots()
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(bottom='off',top='off',left='off',right='off')

ax.plot(np.log10(F_real),np.log10(F_obs),'.', markersize=0.5)
ax.plot(zz,zz,'-')
ax.set_xlim(limit)
ax.set_ylim(limit)

original_xticks1 = ax.get_xticks()
original_yticks1 = ax.get_yticks()

savefig('recalib_scatter_real_vs_obs_points.png')

close() # it closes the previous plot to avoid memory leak

fig, ax = subplots()
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(bottom='off',top='off',left='off',right='off')

ax.plot(np.log10(F_real),np.log10(F_recal),'.', markersize=0.5)
ax.plot(zz,zz,'-')
ax.set_xlim(limit)
ax.set_ylim(limit)

savefig('recalib_scatter_real_vs_recalib_points.png')

close() # it closes the previous plot to avoid memory leak

execfile("rc_settings.py")
rc('figure.subplot', bottom=.2, top=.95, right=.95, left=.25)
rc('figure', figsize=(2.5, 2.5))
if(pdf_format!=True):
  rc('savefig', dpi=100)
  
fig, ax = subplots()

ax.set_xlim(limit)
ax.set_ylim(limit)
ax.set_xlabel('$\log_{10}F_{true}$')
ax.set_ylabel('$\log_{10}F_{noisy}$')

currentdir = os.getcwd()

im = imread(get_sample_data(currentdir+'\\'+'recalib_scatter_real_vs_obs_points.png'))
ax.imshow(im, extent=[original_xticks1[0],original_xticks1[-1],original_yticks1[0], original_yticks1[-1]], aspect="auto")

if(pdf_format):
  savefig('recalib_scatter_real_vs_obs.pdf', format='pdf')
else:
  savefig('recalib_scatter_real_vs_obs.png')

close() # it closes the previous plot to avoid memory leak

fig, ax = subplots()

ax.set_xlim(limit)
ax.set_ylim(limit)
ax.set_xlabel('$\log_{10}F_{true}$')
ax.set_ylabel('$\log_{10}F_{MCMC}$')

currentdir = os.getcwd()

im = imread(get_sample_data(currentdir+'\\'+'recalib_scatter_real_vs_recalib_points.png'))
ax.imshow(im, extent=[original_xticks1[0],original_xticks1[-1],original_yticks1[0], original_yticks1[-1]], aspect="auto")

if(pdf_format):
  savefig('recalib_scatter_real_vs_recalib.pdf', format='pdf')
else:
  savefig('recalib_scatter_real_vs_recalib.png')

t1 = dt.datetime.today()
print 'Elapsed time of generating figures of scatter plots:', t1-t0