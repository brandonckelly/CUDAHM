# executing e.g. python discard_burnin.py lumfunc_thetas_25_06_2016.dat --burn_in_end 2000000
import argparse as argp
import datetime as dt
import numpy as np

parser = argp.ArgumentParser()
parser.add_argument("file", help="The file name of theta data file.", type = str)
parser.add_argument("--burn_in_end", default = 0, help="The end postition of the candidate burn-in period.", type=int)

args = parser.parse_args()
file = args.file
burn_in_end = args.burn_in_end

t0 = dt.datetime.today()

theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))

theta_data_discarded_burnin = theta_data[burn_in_end:,:]

filename = file.split('.')[0]
np.savetxt(filename + '_without_burnin.dat', theta_data_discarded_burnin, fmt='%10.6e')

t1 = dt.datetime.today()
print 'Elapsed time of computation:', t1-t0