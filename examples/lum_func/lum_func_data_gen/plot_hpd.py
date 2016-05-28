# executing e.g. python plot_hpd.py fluxes_cnt_100000.dat filtered_fluxes_w_G_noise_mu_0.0_sig_1.0_fl_5.0_cnt_100000.dat lumfunc_chi_summary_27_05_2016.dat lumfunc_chi_summary_27_05_2016_post_mean_std_dev.dat lumfunc_chi_summary_27_05_2016_hpd_intvals_95.dat  --width 0.95 --pdf_format False
import argparse as argp
import datetime as dt
import numpy as np
from scipy import stats
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("real_flux_file", help="The file name of real flux data file.", type = str)
parser.add_argument("noisy_flux_file", help="The file name of noisy flux data file.", type = str)
parser.add_argument("estimated_flux_file", help="The file name of estimated flux data file.", type = str)
parser.add_argument("post_mean_std_dev_file", help="The file name of estimated flux data file.", type = str)
parser.add_argument("hpd_file", help="The file name of HPD intervals of estimated flux data file.", type = str)
parser.add_argument("--width", default = 0.683, help="Width of the marginal posterior credible region", type=float)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
real_flux_file = args.real_flux_file
noisy_flux_file = args.noisy_flux_file
estimated_flux_file = args.estimated_flux_file
post_mean_std_dev_file = args.post_mean_std_dev_file
hpd_file = args.hpd_file
width = args.width
pdf_format = eval(args.pdf_format)

t0 = dt.datetime.today()

execfile("rc_settings.py")
rc('figure', figsize=(1.9, 1.9))
rc('figure.subplot', bottom=.275, top=.85, right=.85, left=.3)
#rc('figure.subplot', bottom=.2, top=.95, right=.95, left=.28)
#rc('figure', figsize=(2.5, 2.5))
if(pdf_format!=True):
  rc('savefig', dpi=100)

def determineMantExp(floatNum):
    sciNotationPython = '%e' % floatNum
    mantissa, exp = sciNotationPython.split('e')
    return float(mantissa),int(exp)

real_flux_data=np.loadtxt(real_flux_file)
n_samp = len(real_flux_data)

noisy_flux_data=np.loadtxt(noisy_flux_file,delimiter=' ',usecols=(0,1))
noisy_flux_data_0=noisy_flux_data[:,0]
noisy_flux_data_std_dev=noisy_flux_data[:,1]

usecols = tuple(range(n_samp+1))

estimated_flux_data = np.loadtxt(estimated_flux_file,dtype=np.str,delimiter=' ',usecols=usecols)
estimated_flux_data = estimated_flux_data[:,:n_samp].astype(np.float)
post_mean_std_dev_data=np.loadtxt(post_mean_std_dev_file,delimiter=' ',usecols=(0,1))
post_mean_data=post_mean_std_dev_data[:,0]
hpd_intvals=np.loadtxt(hpd_file,delimiter=' ',usecols=(0,1))

# For confidence intervals
z_star = stats.norm.ppf((width+1)/2.0)
print z_star

fluxes_list=[]
for idx in range(0, n_samp):
  confidence_intval_lower = noisy_flux_data_0[idx] - z_star * noisy_flux_data_std_dev[idx]
  confidence_intval_upper = noisy_flux_data_0[idx] + z_star * noisy_flux_data_std_dev[idx]
  fluxes_element=[real_flux_data[idx],noisy_flux_data_0[idx],noisy_flux_data_std_dev[idx],confidence_intval_lower,confidence_intval_upper,post_mean_data[idx],hpd_intvals[idx][0],hpd_intvals[idx][1]]
  for subidx in range(0, estimated_flux_data.shape[0]):
    fluxes_element.append(estimated_flux_data[subidx][idx])
  fluxes_list.append(tuple(fluxes_element))

print fluxes_list[92697]

fluxes = np.array(fluxes_list)

arg_sort = np.argsort(fluxes, axis=0)

F_real = fluxes[arg_sort[:,0],0].flatten()
F_obs = fluxes[arg_sort[:,0],1].flatten()
F_obs_std_dev = fluxes[arg_sort[:,0],2].flatten()
F_obs_conf_intvals = fluxes[arg_sort[:,0],3:5]
F_est_post_mean = fluxes[arg_sort[:,0],5].flatten()
F_est_hpd_intvals = fluxes[arg_sort[:,0],6:8]
F_est = fluxes[arg_sort[:,0],8:]

print F_real[0]
print F_obs[0]
print F_obs_std_dev[0]
print F_obs_conf_intvals[0]
print F_est_post_mean[0]
print F_est_hpd_intvals[0]
print F_est[0]

print F_real[n_samp-1]
print F_obs[n_samp-1]
print F_obs_std_dev[n_samp-1]
print F_obs_conf_intvals[n_samp-1]
print F_est_post_mean[n_samp-1]
print F_est_hpd_intvals[n_samp-1]
print F_est[n_samp-1]

chosen_idxs = [0,int(0.133333*n_samp),int(0.266666*n_samp),int(0.4*n_samp),int(0.7*n_samp),int(0.95*n_samp)]

for obj_idx in chosen_idxs:
  fig, ax = subplots()
  
  ax.axvline(x = F_real[obj_idx], color='red', linewidth=2, linestyle='-',zorder=3)
  ax.axvline(x = F_obs[obj_idx], color='blue', linewidth=2, linestyle='-',zorder=3)
  ax.axvline(x = F_est_post_mean[obj_idx], color='green', linewidth=2, linestyle='-',zorder=3)
  binNO=40 if (obj_idx < 0.8*n_samp) else 10
  n, bins, patches = ax.hist(F_est[obj_idx], bins=40, normed=True, color=(0.0,1.0,0.0), edgecolor=(0.0,1.0,0.0),zorder=1)
  conf_intval_y_loc = 0.05
  ax.plot(F_obs_conf_intvals[obj_idx], [conf_intval_y_loc, conf_intval_y_loc], '-',color='blue',linewidth=2,zorder=2)
  ax.plot([F_obs_conf_intvals[obj_idx][0],F_obs_conf_intvals[obj_idx][0]], [conf_intval_y_loc - 0.025, conf_intval_y_loc + 0.025], '-',color='blue',linewidth=2,zorder=2)
  ax.plot([F_obs_conf_intvals[obj_idx][1],F_obs_conf_intvals[obj_idx][1]], [conf_intval_y_loc - 0.025, conf_intval_y_loc + 0.025], '-',color='blue',linewidth=2,zorder=2)
  hpd_intval_y_loc = 0.95
  ax.plot(F_est_hpd_intvals[obj_idx], [hpd_intval_y_loc, hpd_intval_y_loc], '-',color='green',linewidth=2,zorder=2)
  ax.plot([F_est_hpd_intvals[obj_idx][0],F_est_hpd_intvals[obj_idx][0]], [hpd_intval_y_loc - 0.025, hpd_intval_y_loc + 0.025], '-',color='green',linewidth=2,zorder=2)
  ax.plot([F_est_hpd_intvals[obj_idx][1],F_est_hpd_intvals[obj_idx][1]], [hpd_intval_y_loc - 0.025, hpd_intval_y_loc + 0.025], '-',color='green',linewidth=2,zorder=2)
  
  man, exp = determineMantExp(F_real[obj_idx])
  if exp < 0:
    lbl_title = r"$F_{true}=%10.2f \cdot 10^{%d}$" % (man, exp)
  else:
    lbl_title = r"$F_{true}=%10.2f$" % F_real[obj_idx]
  
  xmin = F_real[obj_idx] if (F_real[obj_idx] < F_est[obj_idx].min()) else F_est[obj_idx].min()
  xmax = max([F_est[obj_idx].max(),F_real[obj_idx],F_obs[obj_idx],F_obs_conf_intvals[obj_idx][1],F_est_hpd_intvals[obj_idx][1]])
  ax.xaxis.set_ticks([xmin,0.5*(xmax+xmin),xmax])
  #ax.xaxis.set_ticklabels([0.0,0.5*F_est[obj_idx].max(),F_est[obj_idx].max()])
  ax.set_xlim([xmin,xmax])
  ax.set_title(lbl_title)
  ax.set_xlabel("$F$")
  ax.set_ylabel("$p(F)$")
  
  if(pdf_format):
    savefig('flux_hpd_' + str(obj_idx) + '.pdf', format='pdf')
  else:
    savefig('flux_hpd_'+ str(obj_idx) + '.png')
  # it closes the previous plot to avoid memory leak:
  close()

t1 = dt.datetime.today()
print 'Elapsed time:', t1-t0