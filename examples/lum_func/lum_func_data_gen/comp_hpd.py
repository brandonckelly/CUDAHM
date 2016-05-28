# executing e.g. python comp_hpd.py lumfunc_chi_summary_27_05_2016.dat 100000 --width 0.95 --pdf_format False
import argparse as argp
import datetime as dt
import numpy as np
from scipy import stats
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("estimated_flux_file", help="The file name of estimated flux data file.", type = str)
parser.add_argument("n_samp", help="The number of objects", type = int)
parser.add_argument("--width", default = 0.683, help="Width of the marginal posterior credible region", type=float)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
estimated_flux_file = args.estimated_flux_file
n_samp = args.n_samp
width = args.width
pdf_format = eval(args.pdf_format)

t0 = dt.datetime.today()

execfile("rc_settings.py")
rc('figure.subplot', bottom=.2, top=.95, right=.95, left=.34)
rc('figure', figsize=(2.5, 2.5))
if(pdf_format!=True):
  rc('savefig', dpi=100)

usecols = tuple(range(n_samp+1))

estimated_flux_data = np.loadtxt(estimated_flux_file,dtype=np.str,delimiter=' ',usecols=usecols)
estimated_flux_data = estimated_flux_data[:,:n_samp].astype(np.float)

def determine_hdp_intval(obj_idx):
  est_flux_data_of_obj_idx = estimated_flux_data[:,obj_idx]
  kernel = stats.gaussian_kde(est_flux_data_of_obj_idx)
  densitiesOfFluxDataOfObjIdx = kernel.evaluate(est_flux_data_of_obj_idx)
  indicesOfSortedDensOfFluxDataOfObjIdx = densitiesOfFluxDataOfObjIdx.argsort()
  minIdx = int(np.ceil((1-width)*len(est_flux_data_of_obj_idx)))
  maxIdx = len(est_flux_data_of_obj_idx) - 1
  flux_of_obj_idx_hpd_lower = float("+inf")
  flux_of_obj_idx_hpd_upper = float("-inf")
  for idx in indicesOfSortedDensOfFluxDataOfObjIdx[minIdx:maxIdx]:
    if est_flux_data_of_obj_idx[idx] < flux_of_obj_idx_hpd_lower:
      flux_of_obj_idx_hpd_lower = est_flux_data_of_obj_idx[idx]
    if est_flux_data_of_obj_idx[idx] > flux_of_obj_idx_hpd_upper:
      flux_of_obj_idx_hpd_upper = est_flux_data_of_obj_idx[idx]
  return flux_of_obj_idx_hpd_lower, flux_of_obj_idx_hpd_upper

output_data = np.zeros((n_samp,2))

for obj_idx in range(n_samp):
  if obj_idx % 10000 == 0:
    print obj_idx
  flux_of_obj_idx_hpd_lower, flux_of_obj_idx_hpd_upper = determine_hdp_intval(obj_idx)
  output_data[obj_idx][0] = flux_of_obj_idx_hpd_lower
  output_data[obj_idx][1] = flux_of_obj_idx_hpd_upper

filename = estimated_flux_file.split('.')[0]
widthstr = str(width).split('.')[1]
np.savetxt(filename + '_hpd_intvals_' + widthstr + '.dat', output_data, fmt='%10.6e')

t1 = dt.datetime.today()
print 'Elapsed time:', t1-t0