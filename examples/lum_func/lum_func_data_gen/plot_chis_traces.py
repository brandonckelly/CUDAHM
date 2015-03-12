# executing e.g. python plot_chis_traces.py lumfunc_chi_min.dat lumfunc_chi_max.dat lumfunc_chi_median.dat b-1.5_l5e10_u5e12_init_b-1.41_l4.0e10_u5.8e12_chi_
import argparse as argp
import numpy as np
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("fileChiMin", help="The file name of chi min data file.", type = str)
parser.add_argument("fileChiMax", help="The file name of chi max data file.", type = str)
parser.add_argument("fileChiMedian", help="The file name of chi median data file.", type = str)
parser.add_argument("prefix", help="The prefic for created output files.", type = str)

args = parser.parse_args()
fileChiMin = args.fileChiMin
fileChiMax = args.fileChiMax
fileChiMedian = args.fileChiMedian
prefix = args.prefix

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

chi_min_data=np.loadtxt(fileChiMin)
chi_max_data=np.loadtxt(fileChiMax)
chi_median_data=np.loadtxt(fileChiMedian)

color_list = range(0, chi_min_data.shape[0])

for idx in range(0, chi_min_data.shape[0]):
    blue_rate = idx/float(chi_min_data.shape[0])
    red_rate = (1.0 - idx/float(chi_min_data.shape[0]))
    color_list[idx]=(red_rate*1.0,0.0,blue_rate*1.0)
    
lbl_min = 'chi min'
lbl_max = 'chi max'
lbl_median = 'chi median'
lbl_iter= 'Iteration number'
	
fig = figure(figsize=(15.75, 10))	
ax = fig.add_subplot(1,1,1) # one row, one column, first plot
ax.scatter(range(1, len(chi_min_data) + 1),chi_min_data, c=color_list, marker = ".", linewidth=0.01)
ax.set_xlabel(lbl_iter)
ax.set_ylabel(lbl_min)
savefig(prefix + 'min.png')

fig = figure(figsize=(15.75, 10))
ax = fig.add_subplot(1,1,1) # one row, one column, first plot
ax.scatter(range(1, len(chi_median_data) + 1),chi_median_data, c=color_list, marker = ".", linewidth=0.01)
ax.set_xlabel(lbl_iter)
ax.set_ylabel(lbl_median)
savefig(prefix + 'median.png')

fig = figure(figsize=(15.75, 10))
ax = fig.add_subplot(1,1,1) # one row, one column, first plot
ax.scatter(range(1, len(chi_max_data) + 1),chi_max_data, c=color_list, marker = ".", linewidth=0.01)
ax.set_xlabel(lbl_iter)
ax.set_ylabel(lbl_max)
savefig(prefix + 'max.png')