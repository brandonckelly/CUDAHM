import numpy as np
from matplotlib.pyplot import *

performance_iter_vs_time_data_1000=np.loadtxt('performance_iter_vs_time_data_1000.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_10000=np.loadtxt('performance_iter_vs_time_data_10000.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_20000=np.loadtxt('performance_iter_vs_time_data_20000.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_50000=np.loadtxt('performance_iter_vs_time_data_50000.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_100000=np.loadtxt('performance_iter_vs_time_data_100000.dat',delimiter=' ',usecols=(0,1))

fig = figure()

ax = fig.add_subplot(1,1,1, xlim=[10000, 1000000], ylim=[0, 150]) # one row, one column, first plot
ax.plot(performance_iter_vs_time_data_1000[:,0], performance_iter_vs_time_data_1000[:,1], 'b-', label = '1000 obj')
ax.plot(performance_iter_vs_time_data_10000[:,0], performance_iter_vs_time_data_10000[:,1], 'r-', label = '10000 obj')
ax.plot(performance_iter_vs_time_data_20000[:,0], performance_iter_vs_time_data_20000[:,1], 'g-', label = '20000 obj')
ax.plot(performance_iter_vs_time_data_50000[:,0], performance_iter_vs_time_data_50000[:,1], 'y-', label = '50000 obj')
ax.plot(performance_iter_vs_time_data_100000[:,0], performance_iter_vs_time_data_100000[:,1], 'm-', label = '100000 obj')

ax.set_xlabel('Iteration numbers')
ax.set_ylabel('Elapsed time (min)')
ax.legend(loc=0)

savefig('performance_iter_vs_time.png',dpi=120)

performance_obj_vs_time_data_10000=np.loadtxt('performance_obj_vs_time_data_10000.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_50000=np.loadtxt('performance_obj_vs_time_data_50000.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_100000=np.loadtxt('performance_obj_vs_time_data_100000.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_500000=np.loadtxt('performance_obj_vs_time_data_500000.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_1000000=np.loadtxt('performance_obj_vs_time_data_1000000.dat',delimiter=' ',usecols=(0,1))

fig = figure()

ax = fig.add_subplot(1,1,1, xlim=[1000, 100000], ylim=[0, 150]) # one row, one column, first plot
ax.plot(performance_obj_vs_time_data_10000[:,0], performance_obj_vs_time_data_10000[:,1], 'b-', label = '10000 iter')
ax.plot(performance_obj_vs_time_data_50000[:,0], performance_obj_vs_time_data_50000[:,1], 'r-', label = '50000 iter')
ax.plot(performance_obj_vs_time_data_100000[:,0], performance_obj_vs_time_data_100000[:,1], 'g-', label = '100000 iter')
ax.plot(performance_obj_vs_time_data_500000[:,0], performance_obj_vs_time_data_500000[:,1], 'y-', label = '500000 iter')
ax.plot(performance_obj_vs_time_data_1000000[:,0], performance_obj_vs_time_data_1000000[:,1], 'm-', label = '1000000 iter')

ax.set_xlabel('Object numbers')
ax.set_ylabel('Elapsed time (min)')
ax.legend(loc=0)

savefig('performance_obj_vs_time.png',dpi=120)