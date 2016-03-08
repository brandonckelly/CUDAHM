# executing e.g. python plot_recalib_quants.py fluxes_cnt_100000.dat filtered_fluxes_w_G_noise_mu_0.0_sig_1.0_fl_5.0_cnt_100000.dat lumfunc_chi_summary_2.dat --pdf_format False
import argparse as argp
import datetime as dt
import numpy as np
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("real_flux_file", help="The file name of real flux data file.", type = str)
parser.add_argument("noisy_flux_file", help="The file name of noisy flux data file.", type = str)
parser.add_argument("estimated_flux_file", help="The file name of estimated flux data file.", type = str)
parser.add_argument("--quants", default = 20, help="The quantile number", type=int)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
real_flux_file = args.real_flux_file
noisy_flux_file = args.noisy_flux_file
estimated_flux_file = args.estimated_flux_file
quants = args.quants
pdf_format = eval(args.pdf_format)

t0 = dt.datetime.today()

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

limit = [0, 2.5]

n = np.int32(F_real.shape[0] / quants)
rms_obs = np.zeros(quants)
rms_recal = np.zeros(quants)
rms_obs_rel = np.zeros(quants)
rms_recal_rel = np.zeros(quants)

for i in range(0, quants):
  fo = F_obs[n*i:n*i+n]
  fr = F_real[n*i:n*i+n]
  fc = F_recal[n*i:n*i+n]
  rms_obs[i] = np.sqrt(np.sum(((fo - fr))**2)/n)
  rms_recal[i] = np.sqrt(np.sum(((fc - fr))**2)/n)
  rms_obs_rel[i] = np.sqrt(np.sum(((fo - fr)/fr)**2)/n)
  rms_recal_rel[i] = np.sqrt(np.sum(((fc - fr)/fr)**2)/n)

bins = np.arange(1, quants + 1)

fig, ax = subplots()
line1, = ax.step(bins,rms_obs, where='mid', label='obs')
line2, = ax.step(bins,rms_recal, where='mid', label='recal')
ax.set_xlabel("quantile")
ax.set_ylabel("RMS abs error")

#legend(loc=2)

if(pdf_format):
  savefig('recalib_quants.pdf', format='pdf')
else:
  savefig('recalib_quants.png')

t1 = dt.datetime.today()
print 'Elapsed time of generating figures of quants plots:', t1-t0