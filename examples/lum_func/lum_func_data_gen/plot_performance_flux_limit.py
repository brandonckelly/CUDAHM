# executing e.g. python plot_performance_flux_limit.py --pdf_format False

import argparse as argp
import numpy as np
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

pdf_format = eval(args.pdf_format)

execfile("rc_settings.py")
rc('font', size=20)  # default for labels (not axis labels)
if(pdf_format!=True):
  rc('savefig', dpi=100)

performance_iter_vs_time_data_1000=np.loadtxt('performance_iter_vs_time_data_1000.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_10000=np.loadtxt('performance_iter_vs_time_data_10000.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_100000=np.loadtxt('performance_iter_vs_time_data_100000.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_300000=np.loadtxt('performance_iter_vs_time_data_300000.dat',delimiter=' ',usecols=(0,1))

fig = figure()

ax = fig.add_subplot(1,1,1, xlim=[0, 1010000], ylim=[0, 60]) # one row, one column, first plot
ax.scatter(performance_iter_vs_time_data_1000[:,0], performance_iter_vs_time_data_1000[:,1], color ='b', marker = "o", s = 15, label = '1000 obj')
ax.plot(performance_iter_vs_time_data_1000[:,0], performance_iter_vs_time_data_1000[:,1], 'b-')
ax.scatter(performance_iter_vs_time_data_10000[:,0], performance_iter_vs_time_data_10000[:,1], color ='m', marker = "o", s = 15, label = '10000 obj')
ax.plot(performance_iter_vs_time_data_10000[:,0], performance_iter_vs_time_data_10000[:,1], 'm-')
ax.scatter(performance_iter_vs_time_data_100000[:,0], performance_iter_vs_time_data_100000[:,1], color ='y', marker = "o", s = 15, label = '100000 obj')
ax.plot(performance_iter_vs_time_data_100000[:,0], performance_iter_vs_time_data_100000[:,1], 'y-')
ax.scatter(performance_iter_vs_time_data_300000[:,0], performance_iter_vs_time_data_300000[:,1], color ='r', marker = "o", s = 15, label = '300000 obj')
ax.plot(performance_iter_vs_time_data_300000[:,0], performance_iter_vs_time_data_300000[:,1], 'r-')

ax.set_xlabel('Iteration numbers')
ax.set_ylabel('Elapsed time (min)')
ax.legend(loc=2)

if(pdf_format):
  savefig('performance_iter_vs_time.pdf', format='pdf')
else:
  savefig('performance_iter_vs_time.png')

performance_obj_vs_time_data_10000=np.loadtxt('performance_obj_vs_time_data_10000.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_20000=np.loadtxt('performance_obj_vs_time_data_20000.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_50000=np.loadtxt('performance_obj_vs_time_data_50000.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_100000=np.loadtxt('performance_obj_vs_time_data_100000.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_200000=np.loadtxt('performance_obj_vs_time_data_200000.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_500000=np.loadtxt('performance_obj_vs_time_data_500000.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_1000000=np.loadtxt('performance_obj_vs_time_data_1000000.dat',delimiter=' ',usecols=(0,1))

fig = figure()

ax = fig.add_subplot(1,1,1, xlim=[0, 303000], ylim=[0, 60]) # one row, one column, first plot

ax.scatter(performance_obj_vs_time_data_10000[:,0], performance_obj_vs_time_data_10000[:,1], color = 'b', marker = "o", s = 15, label = '10000 iter')
ax.plot(performance_obj_vs_time_data_10000[:,0], performance_obj_vs_time_data_10000[:,1], 'b-')
ax.scatter(performance_obj_vs_time_data_20000[:,0], performance_obj_vs_time_data_20000[:,1], color = 'g', marker = "o", s = 15, label = '20000 iter')
ax.plot(performance_obj_vs_time_data_20000[:,0], performance_obj_vs_time_data_20000[:,1], 'g-')
ax.scatter(performance_obj_vs_time_data_50000[:,0], performance_obj_vs_time_data_50000[:,1], color = 'r', marker = "o", s = 15, label = '50000 iter')
ax.plot(performance_obj_vs_time_data_50000[:,0], performance_obj_vs_time_data_50000[:,1], 'r-')
ax.scatter(performance_obj_vs_time_data_100000[:,0], performance_obj_vs_time_data_100000[:,1], color = 'c', marker = "o", s = 15, label = '100000 iter')
ax.plot(performance_obj_vs_time_data_100000[:,0], performance_obj_vs_time_data_100000[:,1], 'c-')
ax.scatter(performance_obj_vs_time_data_200000[:,0], performance_obj_vs_time_data_200000[:,1], color = 'm', marker = "o", s = 15, label = '200000 iter')
ax.plot(performance_obj_vs_time_data_200000[:,0], performance_obj_vs_time_data_200000[:,1], 'm-')
ax.scatter(performance_obj_vs_time_data_500000[:,0], performance_obj_vs_time_data_500000[:,1], color = 'y', marker = "o", s = 15, label = '500000 iter')
ax.plot(performance_obj_vs_time_data_500000[:,0], performance_obj_vs_time_data_500000[:,1], 'y-')
ax.scatter(performance_obj_vs_time_data_1000000[:,0], performance_obj_vs_time_data_1000000[:,1], color = 'k', marker = "o", s = 15, label = '1000000 iter')
ax.plot(performance_obj_vs_time_data_1000000[:,0], performance_obj_vs_time_data_1000000[:,1], 'k-')

ax.set_xlabel('Object numbers')
ax.set_ylabel('Elapsed time (min)')
ax.legend(loc=0)

if(pdf_format):
  savefig('performance_obj_vs_time.pdf', format='pdf')
else:
  savefig('performance_obj_vs_time.png')