# executing e.g. python plot_performance_combined.py --pdf_format False
import argparse as argp
import numpy as np
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
pdf_format = eval(args.pdf_format)

execfile("rc_settings.py")
if(pdf_format!=True):
  rc('savefig', dpi=100)

performance_iter_vs_time_data_10000_without_fl=np.loadtxt('performance_iter_vs_time_data_10000_without_fl.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_100000_without_fl=np.loadtxt('performance_iter_vs_time_data_100000_without_fl.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_300000_without_fl=np.loadtxt('performance_iter_vs_time_data_300000_without_fl.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_600000_without_fl=np.loadtxt('performance_iter_vs_time_data_600000_without_fl.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_1000000_without_fl=np.loadtxt('performance_iter_vs_time_data_1000000_without_fl.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_10000_with_fl=np.loadtxt('performance_iter_vs_time_data_10000_with_fl.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_100000_with_fl=np.loadtxt('performance_iter_vs_time_data_100000_with_fl.dat',delimiter=' ',usecols=(0,1))
performance_iter_vs_time_data_300000_with_fl=np.loadtxt('performance_iter_vs_time_data_300000_with_fl.dat',delimiter=' ',usecols=(0,1))

fig = figure()

ax = fig.add_subplot(1,1,1, xlim=[0, 2020000], ylim=[0, 170]) # one row, one column, first plot
ax.scatter(performance_iter_vs_time_data_10000_without_fl[:,0], performance_iter_vs_time_data_10000_without_fl[:,1], color ='b', marker = "o", s = 15, label = '10000 objects')
ax.plot(performance_iter_vs_time_data_10000_without_fl[:,0], performance_iter_vs_time_data_10000_without_fl[:,1], 'b-')
ax.scatter(performance_iter_vs_time_data_100000_without_fl[:,0], performance_iter_vs_time_data_100000_without_fl[:,1], color ='g', marker = "o", s = 15, label = '100000 objects')
ax.plot(performance_iter_vs_time_data_100000_without_fl[:,0], performance_iter_vs_time_data_100000_without_fl[:,1], 'g-')
ax.scatter(performance_iter_vs_time_data_300000_without_fl[:,0], performance_iter_vs_time_data_300000_without_fl[:,1], color ='r', marker = "o", s = 15, label = '300000 objects')
ax.plot(performance_iter_vs_time_data_300000_without_fl[:,0], performance_iter_vs_time_data_300000_without_fl[:,1], 'r-')
ax.scatter(performance_iter_vs_time_data_600000_without_fl[:,0], performance_iter_vs_time_data_600000_without_fl[:,1], color ='gray', marker = "o", s = 15, label = '600000 objects')
ax.plot(performance_iter_vs_time_data_600000_without_fl[:,0], performance_iter_vs_time_data_600000_without_fl[:,1], color='gray', linestyle='solid')
ax.scatter(performance_iter_vs_time_data_1000000_without_fl[:,0], performance_iter_vs_time_data_1000000_without_fl[:,1], color ='gray', marker = "o", s = 15, label = '1000000 objects')
ax.plot(performance_iter_vs_time_data_1000000_without_fl[:,0], performance_iter_vs_time_data_1000000_without_fl[:,1], color='gray', linestyle='solid')
ax.scatter(performance_iter_vs_time_data_10000_with_fl[:,0], performance_iter_vs_time_data_10000_with_fl[:,1], color ='b', marker = "o", s = 15)
ax.plot(performance_iter_vs_time_data_10000_with_fl[:,0], performance_iter_vs_time_data_10000_with_fl[:,1], 'b--')
ax.scatter(performance_iter_vs_time_data_100000_with_fl[:,0], performance_iter_vs_time_data_100000_with_fl[:,1], color ='g', marker = "o", s = 15)
ax.plot(performance_iter_vs_time_data_100000_with_fl[:,0], performance_iter_vs_time_data_100000_with_fl[:,1], 'g--')
ax.scatter(performance_iter_vs_time_data_300000_with_fl[:,0], performance_iter_vs_time_data_300000_with_fl[:,1], color ='r', marker = "o", s = 15)
ax.plot(performance_iter_vs_time_data_300000_with_fl[:,0], performance_iter_vs_time_data_300000_with_fl[:,1], 'r--')

#The following 'custom legend' based on http://stackoverflow.com/questions/13303928/how-to-make-custom-legend-in-matplotlib
#Get artists and labels for legend and chose which ones to display
handles, labels = ax.get_legend_handles_labels()
display = tuple(range(5))

#Create custom artists
withoutFLArtist = Line2D((0,1),(0,0), color='k', linestyle='-')
withFLArtist = Line2D((0,1),(0,0), color='k', linestyle='--')

#Create legend from custom artist/label lists
ax.legend([handle for i,handle in enumerate(handles) if i in display]+[withoutFLArtist,withFLArtist],
          [label for i,label in enumerate(labels) if i in display]+['without flux limit', 'with flux limit 0.5'], loc=2)

ax.set_xlabel('Iteration number')
ax.set_ylabel('Elapsed time (min)')
#ax.legend(loc=2)

if(pdf_format):
  savefig('performance_iter_vs_time.pdf', format='pdf')
else:
  savefig('performance_iter_vs_time.png')

performance_obj_vs_time_data_10000_without_fl=np.loadtxt('performance_obj_vs_time_data_10000_without_fl.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_200000_without_fl=np.loadtxt('performance_obj_vs_time_data_200000_without_fl.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_500000_without_fl=np.loadtxt('performance_obj_vs_time_data_500000_without_fl.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_1000000_without_fl=np.loadtxt('performance_obj_vs_time_data_1000000_without_fl.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_2000000_without_fl=np.loadtxt('performance_obj_vs_time_data_2000000_without_fl.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_10000_with_fl=np.loadtxt('performance_obj_vs_time_data_10000_with_fl.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_200000_with_fl=np.loadtxt('performance_obj_vs_time_data_200000_with_fl.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_500000_with_fl=np.loadtxt('performance_obj_vs_time_data_500000_with_fl.dat',delimiter=' ',usecols=(0,1))
performance_obj_vs_time_data_1000000_with_fl=np.loadtxt('performance_obj_vs_time_data_1000000_with_fl.dat',delimiter=' ',usecols=(0,1))

fig = figure()

ax = fig.add_subplot(1,1,1, xlim=[0, 1010000], ylim=[0, 170]) # one row, one column, first plot
ax.scatter(performance_obj_vs_time_data_10000_without_fl[:,0], performance_obj_vs_time_data_10000_without_fl[:,1], color = 'b', marker = "o", s = 15, label = '10000 iterations')
ax.plot(performance_obj_vs_time_data_10000_without_fl[:,0], performance_obj_vs_time_data_10000_without_fl[:,1], 'b-')
ax.scatter(performance_obj_vs_time_data_200000_without_fl[:,0], performance_obj_vs_time_data_200000_without_fl[:,1], color = 'g', marker = "o", s = 15, label = '200000 iterations')
ax.plot(performance_obj_vs_time_data_200000_without_fl[:,0], performance_obj_vs_time_data_200000_without_fl[:,1], 'g-')
ax.scatter(performance_obj_vs_time_data_500000_without_fl[:,0], performance_obj_vs_time_data_500000_without_fl[:,1], color = 'r', marker = "o", s = 15, label = '500000 iterations')
ax.plot(performance_obj_vs_time_data_500000_without_fl[:,0], performance_obj_vs_time_data_500000_without_fl[:,1], 'r-')
ax.scatter(performance_obj_vs_time_data_1000000_without_fl[:,0], performance_obj_vs_time_data_1000000_without_fl[:,1], color = 'm', marker = "o", s = 15, label = '1000000 iterations')
ax.plot(performance_obj_vs_time_data_1000000_without_fl[:,0], performance_obj_vs_time_data_1000000_without_fl[:,1], 'm-')
ax.scatter(performance_obj_vs_time_data_2000000_without_fl[:,0], performance_obj_vs_time_data_2000000_without_fl[:,1], color = 'gray', marker = "o", s = 15, label = '2000000 iterations')
ax.plot(performance_obj_vs_time_data_2000000_without_fl[:,0], performance_obj_vs_time_data_2000000_without_fl[:,1], color='gray', linestyle='solid')
ax.scatter(performance_obj_vs_time_data_10000_with_fl[:,0], performance_obj_vs_time_data_10000_with_fl[:,1], color = 'b', marker = "o", s = 15)
ax.plot(performance_obj_vs_time_data_10000_with_fl[:,0], performance_obj_vs_time_data_10000_with_fl[:,1], 'b--')
ax.scatter(performance_obj_vs_time_data_200000_with_fl[:,0], performance_obj_vs_time_data_200000_with_fl[:,1], color = 'g', marker = "o", s = 15)
ax.plot(performance_obj_vs_time_data_200000_with_fl[:,0], performance_obj_vs_time_data_200000_with_fl[:,1], 'g--')
ax.scatter(performance_obj_vs_time_data_500000_with_fl[:,0], performance_obj_vs_time_data_500000_with_fl[:,1], color = 'r', marker = "o", s = 15)
ax.plot(performance_obj_vs_time_data_500000_with_fl[:,0], performance_obj_vs_time_data_500000_with_fl[:,1], 'r--')
ax.scatter(performance_obj_vs_time_data_1000000_with_fl[:,0], performance_obj_vs_time_data_1000000_with_fl[:,1], color = 'm', marker = "o", s = 15)
ax.plot(performance_obj_vs_time_data_1000000_with_fl[:,0], performance_obj_vs_time_data_1000000_with_fl[:,1], 'm--')

#The following 'custom legend' based on http://stackoverflow.com/questions/13303928/how-to-make-custom-legend-in-matplotlib
#Get artists and labels for legend and chose which ones to display
handles, labels = ax.get_legend_handles_labels()
display = tuple(range(6))

#Create custom artists
withoutFLArtist = Line2D((0,1),(0,0), color='k', linestyle='-')
withFLArtist = Line2D((0,1),(0,0), color='k', linestyle='--')

#Create legend from custom artist/label lists
ax.legend([handle for i,handle in enumerate(handles) if i in display]+[withoutFLArtist,withFLArtist],
          [label for i,label in enumerate(labels) if i in display]+['without flux limit', 'with flux limit 0.5'], loc=2)

ax.set_xlabel('Object number')
ax.set_ylabel('Elapsed time (min)')
#ax.legend(loc=0)
if(pdf_format):
  savefig('performance_obj_vs_time.pdf', format='pdf')
else:
  savefig('performance_obj_vs_time.png')