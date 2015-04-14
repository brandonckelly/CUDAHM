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

rc('figure', figsize=(16.5,10.5)) # default figure size in inches (8,6)

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
#fig.set_size_inches(16.5,10.5)
rects1 = ax.bar(ind, mse_max_like_list, width, color='b')
rects2 = ax.bar(ind+width, mse_post_mean_list, width, color='r')

# add some text for labels, title and axes ticks
ax.set_ylabel('Mean Squared Error')
ax.set_title('Comparison Mean Squared Error by Different Regimes')
ax.set_xticks(ind+width)
ax.set_xticklabels( regimes_text_list )

# make font size of tick labels smaller and rotate it vertically:
# based on http://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller/11386056#11386056
for tick in ax.xaxis.get_major_ticks():
  tick.label.set_fontsize(10)
  tick.label.set_rotation('vertical')

ax.legend( (rects1[0], rects2[0]), ('Max like', 'Post mean') )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

#legend(loc=0)
savefig('comp_mse_for_max_like_and_post_mean.png')