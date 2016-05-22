# executing e.g. python comp_hpd.py lumfunc_chi_summary_2.dat --rate 0.05 --width 0.683 --pdf_format False
import argparse as argp
import datetime as dt
import numpy as np
from scipy import stats
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("estimated_flux_file", help="The file name of estimated flux data file.", type = str)
parser.add_argument("--rate", default = 0.01, help="Rate of samples which is included for the computation of the kernel estimate", type=float)
parser.add_argument("--width", default = 0.95, help="Width of the marginal posterior credible region", type=float)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
real_flux_file = args.real_flux_file
noisy_flux_file = args.noisy_flux_file
estimated_flux_file = args.estimated_flux_file
rate = args.rate
width = args.width
pdf_format = eval(args.pdf_format)

t0 = dt.datetime.today()

execfile("rc_settings.py")
rc('figure.subplot', bottom=.2, top=.95, right=.95, left=.34)
rc('figure', figsize=(2.5, 2.5))
if(pdf_format!=True):
  rc('savefig', dpi=100)

estimated_flux_data=np.loadtxt(estimated_flux_file,delimiter=' ',usecols=(0,1))
estimated_flux_data_0=estimated_flux_data[:,0]

n_est_flux_data_for_kernel = int(rate*len(estimated_flux_data_0))
est_flux_data_for_kernel = estimated_flux_data_0[:n_est_flux_data_for_kernel]
print n_est_flux_data_for_kernel
kernel = stats.gaussian_kde(est_flux_data_for_kernel)
densitiesOfPoints = kernel.evaluate(estimated_flux_data_0)
for idx in range(0,10):
  print "x:",estimated_flux_data_0[idx],"density:",densitiesOfPoints[idx]

indicesOfSortedDensOfPoints = densitiesOfPoints.argsort()
for idx in range(0,10):
  print "x:",estimated_flux_data_0[indicesOfSortedDensOfPoints[idx]],"density:",densitiesOfPoints[indicesOfSortedDensOfPoints[idx]]

minIdx = int(np.ceil((1-width)*len(estimated_flux_data_0)))
maxIdx = len(estimated_flux_data_0) - 1
print minIdx,maxIdx
flux_hpd_lower = float("+inf")
flux_hpd_upper = float("-inf")
plotted_data_list = []
for idx in indicesOfSortedDensOfPoints[minIdx:maxIdx]:
  plotted_data_list.append((estimated_flux_data_0[idx],densitiesOfPoints[idx]))
  if estimated_flux_data_0[idx] < flux_hpd_lower:
    flux_hpd_lower = estimated_flux_data_0[idx]
  if estimated_flux_data_0[idx] > flux_hpd_upper:
    flux_hpd_upper = estimated_flux_data_0[idx]

hpd_str = "Highest posterior density (HPD) region: [%5.4f,%5.4f]" % (flux_hpd_lower,flux_hpd_upper)
print hpd_str

plotted_data = np.array(plotted_data_list)
arg_sort = np.argsort(plotted_data, axis=0)
xdata = plotted_data[arg_sort[:,0],0].flatten()
ydata = plotted_data[arg_sort[:,0],1].flatten()

fig, ax = subplots()
ax.plot(xdata, ydata, '-', color='green')
ax.set_xlabel("$F_{MCMC}$")
ax.set_ylabel("$p(F_{MCMC})$")

ax.set_xlim([flux_hpd_lower,flux_hpd_upper])
#ax.set_ylim([densitiesOfPoints.min(), densitiesOfPoints.max()])

if(pdf_format):
  savefig('recalib_density.pdf', format='pdf')
else:
  savefig('recalib_density.png')

t1 = dt.datetime.today()
print 'Elapsed time:', t1-t0