# executing e.g. python plot_pairs_plot.py lumfunc_thetas_2.dat _ --lower_scale_factor 10000000000.0 --upper_scale_factor 1000000000000.0 --smooth 0.85 --pdf_format False
import argparse as argp
import datetime as dt
import numpy as np
from matplotlib.pyplot import *
import corner

parser = argp.ArgumentParser()
parser.add_argument("file", help="The file name of theta data file.", type = str)
parser.add_argument("prefix", help="The prefic for created output files.", type = str)
parser.add_argument("--lower_scale_factor", default = 1.0, help="The factor which scales up the lower scale samples", type=float)
parser.add_argument("--upper_scale_factor", default = 1.0, help="The factor which scales up the upper scale samples", type=float)
parser.add_argument("--smooth", default = 0.0, help="The standard deviation for Gaussian kernel passed to gaussian_filter of corner method", type=float)
parser.add_argument("--bins", default = 50, help="The bin numbers of histogram", type=int)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
file = args.file
prefix = args.prefix
lower_scale_factor = args.lower_scale_factor
upper_scale_factor = args.upper_scale_factor
smooth = args.smooth
bins = args.bins
pdf_format = eval(args.pdf_format)

t0 = dt.datetime.today()

execfile("rc_settings.py")

def determineMagnitude(floatNum):
    sciNotationPython = '%e' % floatNum
    mantissa, exp = sciNotationPython.split('e+')
    return 10.0**int(exp),int(exp)

theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))

max_lowerscale = theta_data[:,1].max()*lower_scale_factor
max_lowerscale_mag, max_lowerscale_exp = determineMagnitude(max_lowerscale)
max_upperscale = theta_data[:,2].max()*upper_scale_factor
max_upperscale_mag, max_upperscale_exp = determineMagnitude(max_upperscale)

lbl_lowerscale = r'$l$ ($\times 10^{%d}$)' % max_lowerscale_exp
lbl_upperscale = r'$u$ ($\times 10^{%d}$)' % max_upperscale_exp
figure = corner.corner(theta_data, labels=[r'$\beta$', lbl_lowerscale, lbl_upperscale], color='b', bins=bins, smooth=smooth, hist_kwargs={'color':'b','edgecolor':'b','zorder':5})

if(pdf_format):
  savefig('pairs_plot.pdf', format='pdf')
else:
  savefig('pairs_plot.png')
  
t1 = dt.datetime.today()
print 'Elapsed time of generating figures of error plots:', t1-t0