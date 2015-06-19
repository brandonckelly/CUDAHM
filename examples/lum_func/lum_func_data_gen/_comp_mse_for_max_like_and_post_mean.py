# executing e.g. python _comp_mse_for_max_like_and_post_mean.py fluxes_cnt_100000.dat filtered_fluxes_w_G_noise_mu_0.0_sig_1.0_fl_5.0_cnt_100000.dat lumfunc_chi_summary.dat

import argparse as argp
import datetime as dt
import numpy as np
from matplotlib.pyplot import *

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

execfile("rc_settings.py")

# Wider margins to allow for larger labels; may need to adjust left:
# This is a different setting because the original bottom is .125
rc('figure.subplot', bottom=.175, top=.95, right=.95)  # left=0.125

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

rate=len(sorted_fluxes)/n_regimes

mse_max_like_list = []
mse_post_mean_list = []

regimes_text_list = []

for idx in range(1,n_regimes+1):
  lim_reg_lower = rate * (idx - 1)
  lim_reg_upper = rate * idx
  current_mse_max_like = 0.0
  current_mse_post_mean = 0.0
  current_text = '['+str(lim_reg_lower)+':'+str(lim_reg_upper-1)+']'
  regimes_text_list.append(current_text)
  for subIdx in range(lim_reg_lower,lim_reg_upper):
    current_mse_max_like += (sorted_fluxes[subIdx][1] - sorted_fluxes[subIdx][0])**2.0
    current_mse_post_mean += (sorted_fluxes[subIdx][2] - sorted_fluxes[subIdx][0])**2.0
  current_mse_max_like /= rate
  current_mse_post_mean /= rate
  mse_max_like_list.append(current_mse_max_like)
  mse_post_mean_list.append(current_mse_post_mean)
  
print "MSE of max like - true", mse_max_like_list
print "MSE of post mean - true", mse_post_mean_list

# based on matplotlib example: http://matplotlib.org/examples/api/barchart_demo.html

ind = np.arange(n_regimes)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = subplots()

ax.set_xlim([-0.3333, n_regimes])
ax.set_ylim([0, max(max(mse_max_like_list), max(mse_post_mean_list))*1.25])

rects1 = ax.bar(ind, mse_max_like_list, width, color='b', label = 'Max like')
rects2 = ax.bar(ind+width, mse_post_mean_list, width, color='r', label = 'Post mean')

# add some text for labels, title and axes ticks
ax.set_ylabel('Mean Squared Errors')
ax.set_title('Comparison Mean Squared Errors by Different Regimes')
ax.set_xticks(ind+width)
ax.set_xticklabels( regimes_text_list )

# make font size of tick labels smaller and rotate it vertically:
# based on http://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller/11386056#11386056
for tick in ax.xaxis.get_major_ticks():
  tick.label.set_rotation('vertical')

ax.legend(loc=2)  # upper left

# based on: http://matplotlib.org/users/legend_guide.html#legend-location
#ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # Place a legend to the right of this smaller figure.

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 0.5 + height, '%5.5f'%float(height),
                ha='center', va='bottom', size='small', rotation = 'vertical')

autolabel(rects1)
autolabel(rects2)

#legend(loc=0)
savefig('comp_mse_for_max_like_and_post_mean.png')