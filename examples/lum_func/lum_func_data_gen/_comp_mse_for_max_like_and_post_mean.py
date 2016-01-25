# executing e.g. python _comp_mse_for_max_like_and_post_mean.py fluxes_cnt_100000.dat filtered_fluxes_w_G_noise_mu_0.0_sig_1.0_fl_5.0_cnt_100000.dat lumfunc_chi_summary_2.dat --pdf_format False
import argparse as argp
import datetime as dt
import numpy as np
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("real_flux_file", help="The file name of real flux data file.", type = str)
parser.add_argument("noisy_flux_file", help="The file name of noisy flux data file.", type = str)
parser.add_argument("estimated_flux_file", help="The file name of estimated flux data file.", type = str)
parser.add_argument("--n_regions", default = 10, help="The number of flux regions.", type = int)
parser.add_argument("--broken_y_axis_start", default = 0.16, help="The start position of y-axis breaking.", type = float)
parser.add_argument("--broken_y_axis_end", default = 0.25, help="The end position of y-axis breaking.", type = float)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
real_flux_file = args.real_flux_file
noisy_flux_file = args.noisy_flux_file
estimated_flux_file = args.estimated_flux_file
n_regions = args.n_regions
broken_y_axis_start = args.broken_y_axis_start
broken_y_axis_end = args.broken_y_axis_end
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
sorted_fluxes = np.sort(fluxes,axis=0)

print sorted_fluxes

rate=len(sorted_fluxes)/n_regions

mse_max_like_list = []
mse_post_mean_list = []

regions_text_list = []
def indexToOrdinal(idx):
  if(idx==1):
    retVal = r'$1^{st}$'
  elif(idx==2):
    retVal = r'$2^{nd}$'
  elif(idx==3):
    retVal = r'$3^{rd}$'
  else:
    retVal = r'$' + str(idx) + r'^{th}$'
  return retVal

for idx in range(1,n_regions+1):
  lim_reg_lower = rate * (idx - 1)
  lim_reg_upper = rate * idx
  current_mse_max_like = 0.0
  current_mse_post_mean = 0.0
  current_text = indexToOrdinal(idx)
  regions_text_list.append(current_text)
  for subIdx in range(lim_reg_lower,lim_reg_upper):
    current_mse_max_like += (sorted_fluxes[subIdx][1] - sorted_fluxes[subIdx][0])**2.0
    current_mse_post_mean += (sorted_fluxes[subIdx][2] - sorted_fluxes[subIdx][0])**2.0
  current_mse_max_like /= rate
  current_mse_post_mean /= rate
  mse_max_like_list.append(current_mse_max_like)
  mse_post_mean_list.append(current_mse_post_mean)
  
print "MSE of max like - true", mse_max_like_list
print "MSE of post mean - true", mse_post_mean_list

#We need to create broken axis because of outlier values (based on matplotlib example: http://matplotlib.org/examples/pylab_examples/broken_axis.html)
fig, (ax, ax2) = subplots(2, 1, sharex=True)

ax.set_xlim([-0.3333, n_regions])
ax2.set_xlim([-0.3333, n_regions])
ax.set_ylim([broken_y_axis_end, max(max(mse_max_like_list), max(mse_post_mean_list))*1.03])
ax2.set_ylim([0, broken_y_axis_start])

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# based on matplotlib example: http://matplotlib.org/examples/api/barchart_demo.html
ind = np.arange(n_regions)  # the x locations for the groups
width = 0.35       # the width of the bars
rects1 = ax.bar(ind, mse_max_like_list, width, color='b', label = 'Max like')
rects2 = ax.bar(ind+width, mse_post_mean_list, width, color='r', label = 'Post mean')
rects1 = ax2.bar(ind, mse_max_like_list, width, color='b')
rects2 = ax2.bar(ind+width, mse_post_mean_list, width, color='r')

# add some text for labels, title and axes ticks
ax2.set_xlabel('Pieces of flux range')
# For the 'common' y-axis label:
fig.text(0.04, 0.5, 'Mean Squared Errors', verticalalignment='center', rotation='vertical', family='Computer Modern Roman', fontsize=12)
#ax.set_title('Comparison Mean Squared Errors by Different Regimes')
ax2.set_xticks(ind+width)
ax2.set_xticklabels( regions_text_list )

ax.legend(loc=2)  # upper left

if(pdf_format):
  savefig('comp_mse_for_max_like_and_post_mean.pdf', format='pdf')
else:
  savefig('comp_mse_for_max_like_and_post_mean.png')