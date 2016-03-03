# executing e.g. python plot_rms_abs_error.py fluxes_cnt_100000.dat filtered_fluxes_w_G_noise_mu_0.0_sig_1.0_fl_5.0_cnt_100000.dat lumfunc_chi_summary_2.dat --pdf_format False
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

execfile("rc_settings.py")
if(pdf_format!=True):
  rc('savefig', dpi=100)

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

F_real = fluxes[arg_sort,0].flatten()
F_obs = fluxes[arg_sort,1].flatten()
F_recal = fluxes[arg_sort,2].flatten()

logF_real = np.log10(F_real)
hist, bins = np.histogram(logF_real,bins=100)

limit = [0, 2.5]

rms_obs = np.zeros(hist.shape)
rms_recal = np.zeros(hist.shape)
rms_obs_rel = np.zeros(hist.shape)
rms_recal_rel = np.zeros(hist.shape)

for i in range(0, hist.shape[0]):
  filt = bins[i] <= logF_real
  filt = np.logical_and(filt, logF_real < bins[i + 1])
  fo = F_obs[filt]
  fr = F_real[filt]
  fc = F_recal[filt]
  n = sum(filt)
  if n > 0:
    rms_obs[i] = np.sqrt(np.sum(((fo - fr))**2)/n)
    rms_recal[i] = np.sqrt(np.sum(((fc - fr))**2)/n)
    rms_obs_rel[i] = np.sqrt(np.sum(((fo - fr)/fr)**2)/n)
    rms_recal_rel[i] = np.sqrt(np.sum(((fc - fr)/fr)**2)/n)
  else:
    rms_obs[i] = 0
    rms_recal[i] = 0
    rms_obs_rel[i] = 0
    rms_recal_rel[i] = 0

fig, ax = subplots()
line1, = ax.plot(bins[:-1],rms_obs, '-', label='obs')
line2, = ax.plot(bins[:-1],rms_recal, '-', label='recal')
ax.set_xlabel("$\log_{10}F_{real}$")
ax.set_ylabel("RMS abs error")

ax.set_xlim(limit)
ax.set_ylim([0, 5])
#legend(handles=[line1, line2])
legend(loc=4) # lower right

if(pdf_format):
  savefig('rms_abs_error.pdf', format='pdf')
else:
  savefig('rms_abs_error.png')