# executing e.g. python comp_post_mean_std_dev.py lumfunc_chi_summary_27_05_2016.dat 100000
import argparse as argp
import datetime as dt
import numpy as np

parser = argp.ArgumentParser()
parser.add_argument("estimated_flux_file", help="The file name of estimated flux data file.", type = str)
parser.add_argument("n_samp", help="The number of objects", type = int)

args = parser.parse_args()
estimated_flux_file = args.estimated_flux_file
n_samp = args.n_samp

t0 = dt.datetime.today()

usecols = tuple(range(n_samp+1))

estimated_flux_data=np.loadtxt(estimated_flux_file,dtype=np.str,delimiter=' ',usecols=usecols)
estimated_flux_data=estimated_flux_data[:,:n_samp].astype(np.float)

output_data = np.zeros((n_samp,2))

for obj_idx in range(n_samp):
  if obj_idx % 10000 == 0:
    print obj_idx
  post_mean = 0.0
  post_msqr = 0.0
  for iter_idx in range(estimated_flux_data.shape[0]):
    post_mean += estimated_flux_data[iter_idx][obj_idx]
    post_msqr += estimated_flux_data[iter_idx][obj_idx] * estimated_flux_data[iter_idx][obj_idx]
  post_mean /= estimated_flux_data.shape[0]
  post_msqr /= estimated_flux_data.shape[0]
  post_sigma = np.sqrt(post_msqr - post_mean * post_mean)
  output_data[obj_idx][0] = post_mean
  output_data[obj_idx][1] = post_sigma
    
filename = estimated_flux_file.split('.')[0]
np.savetxt(filename + '_post_mean_std_dev.dat', output_data, fmt='%10.6e')

t1 = dt.datetime.today()
print 'Elapsed time:', t1-t0