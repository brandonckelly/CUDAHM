# executing e.g. python plot_thetas_histo.py lumfunc_thetas_2.dat --lower_scale_factor 10000000000.0 --upper_scale_factor 1000000000000.0
import argparse as argp
import datetime as dt
import numpy as np
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("file", help="The file name of theta data file.", type = str)
parser.add_argument("--lower_scale_factor", default = 1.0, help="The factor which scales up the lower scale samples", type=float)
parser.add_argument("--upper_scale_factor", default = 1.0, help="The factor which scales up the upper scale samples", type=float)
parser.add_argument("--nthin_theta", default = 1.0, help="The thinning factor for theta", type=float)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
file = args.file
lower_scale_factor = args.lower_scale_factor
upper_scale_factor = args.upper_scale_factor
nthin_theta = args.nthin_theta
pdf_format = eval(args.pdf_format)

execfile("rc_settings.py")
rc('figure', figsize=(1.9, 1.9))
rc('figure.subplot', bottom=.275, top=.85, right=.85, left=.3)
if(pdf_format!=True):
  rc('savefig', dpi=100)

t0 = dt.datetime.today()

# Max-like (beta,l,u): (-1.5564, 7.3222\cdot 10^{10}, 5.7207\cdot 10^{12})

beta_min = -1.56
beta_max = -1.44

l_min = 2.6 * 10**10
l_max = 7.4 * 10**10

u_min = 4.2 * 10**12
u_max = 5.8 * 10**12

theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))
theta_data_beta=theta_data[:,0]
theta_data_l=theta_data[:,1]
theta_data_u=theta_data[:,2]

theta_data_l *= lower_scale_factor
theta_data_u *= upper_scale_factor

print "Beta min:", theta_data_beta.min(), "max:", theta_data_beta.max()
print "Lowerscale min:", theta_data_l.min(), "max:", theta_data_l.max()
print "Upperscale min:", theta_data_u.min(), "max:", theta_data_u.max()

fig, ax = subplots()
#xlin = np.linspace(theta_data_beta.min(), theta_data_beta.max(), 300)
xlin = np.linspace(beta_min, beta_max, 200)
n, bins, patches = ax.hist(theta_data_beta, bins=xlin, log=False, normed=False, color=(0.8,1.0,0.8), edgecolor=(0.8,1.0,0.8), zorder=1)
#print n
#print bins
#print len(bins) # 300
print n.max(),n.argmax()

# We emphasize that bin which contains the most of the elements:
ax.bar(bins[n.argmax()], n.max(), width =  bins[n.argmax()+1] - bins[n.argmax()], bottom = 0.0, color=(0.0,1.0,0.0), edgecolor=(0.0,1.0,0.0), zorder=3)

# We represent the beta value from maximum likelihood estimation:
ax.axvline(x = -1.5564, color='blue', linewidth=1.0, linestyle='-', zorder=2)

# We represent the true beta value:
ax.axvline(x = -1.5, color='red', linewidth=1.0, linestyle='-', zorder=2)

ax.set_xlim([beta_min, beta_max])
ax.xaxis.set_ticks([beta_min,-1.5,beta_max])
ax.yaxis.set_ticks([0,150,300,450])

ax.set_title(r'$\beta$')

if(pdf_format):
  savefig('beta_histo.pdf', format='pdf')
else:
  savefig('beta_histo.png')
  
fig, ax = subplots()
xlin = np.linspace(l_min, l_max, 200)
n, bins, patches = ax.hist(theta_data_l, bins=xlin, log=False, normed=False, color=(0.8,1.0,0.8), edgecolor=(0.8,1.0,0.8), zorder=1)
print n.max(),n.argmax()

# We emphasize that bin which contains the most of the elements:
ax.bar(bins[n.argmax()], n.max(), width =  bins[n.argmax()+1] - bins[n.argmax()], bottom = 0.0, color=(0.0,1.0,0.0), edgecolor=(0.0,1.0,0.0), zorder=3)

# We represent the lowerscale value from maximum likelihood estimation:
ax.axvline(x = 7.3222 * 10**10, color='blue', linewidth=1.0, linestyle='-', zorder=2)

# We represent the true lowerscale value:
ax.axvline(x = 5.0 * 10**10, color='red', linewidth=1.0, linestyle='-', zorder=2)

ax.set_xlim([l_min, l_max])
ax.xaxis.set_ticks([l_min,5.0*10**10,l_max])
ax.yaxis.set_ticks([0,110,220,330])

ax.set_title(r'$l$')

if(pdf_format):
  savefig('lowerscale_histo.pdf', format='pdf')
else:
  savefig('lowerscale_histo.png')
  
fig, ax = subplots()
xlin = np.linspace(u_min, u_max, 200)
n, bins, patches = ax.hist(theta_data_u, bins=xlin, log=False, normed=False, color=(0.8,1.0,0.8), edgecolor=(0.8,1.0,0.8), zorder=1)
print n.max(),n.argmax()

# We emphasize that bin which contains the most of the elements:
ax.bar(bins[n.argmax()], n.max(), width =  bins[n.argmax()+1] - bins[n.argmax()], bottom = 0.0, color=(0.0,1.0,0.0), edgecolor=(0.0,1.0,0.0), zorder=3)

# We represent the upperscale value from maximum likelihood estimation:
ax.axvline(x = 5.7207 * 10**12, color='blue', linewidth=1.0, linestyle='-', zorder=2)

# We represent the true upperscale value:
ax.axvline(x = 5.0 * 10**12, color='red', linewidth=1.0, linestyle='-', zorder=2)

ax.set_xlim([u_min, u_max])
ax.xaxis.set_ticks([u_min,5.0*10**12,u_max])
ax.yaxis.set_ticks([0,250,500,750,1000])

ax.set_title(r'$u$')

if(pdf_format):
  savefig('upperscale_histo.pdf', format='pdf')
else:
  savefig('upperscale_histo.png')

t1 = dt.datetime.today()
print 'Elapsed time of generating figure of thetas histograms:', t1-t0