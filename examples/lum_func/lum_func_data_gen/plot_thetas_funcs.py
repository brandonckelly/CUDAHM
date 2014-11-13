# executing e.g. python plot_thetas_funcs.py
import numpy as np
from matplotlib.pyplot import *

# Use TeX labels with CMR font:
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

theta_data=np.loadtxt('lumfunc_thetas.dat',delimiter=' ',usecols=(0,1,2))
theta_data_beta=theta_data[:,0]
theta_data_l=theta_data[:,1]
theta_data_u=theta_data[:,2]

color_list = range(0, theta_data.shape[0])

for idx in range(0, theta_data.shape[0]):
    red_rate = (1.0 - idx/float(theta_data.shape[0]))
    color_list[idx]=(red_rate*1.0,1.0,0.0)
    
lbl_beta = r'$\beta$'
lbl_lowerscale = 'lower scale'
lbl_upperscale = 'upper scale'	
	
fig = figure(figsize=(15.75, 10))	
ax = fig.add_subplot(1,1,1) # one row, one column, first plot
ax.scatter(theta_data_beta,theta_data_l, c=color_list)
ax.set_xlabel(lbl_beta)
ax.set_ylabel(lbl_lowerscale)
savefig('beta_lowerscale.png',dpi=120)

fig = figure(figsize=(15.75, 10))
ax = fig.add_subplot(1,1,1) # one row, one column, first plot
ax.scatter(theta_data_beta,theta_data_u, c=color_list)
ax.set_xlabel(lbl_beta)
ax.set_ylabel(lbl_upperscale)
savefig('beta_upperscale.png',dpi=120)

fig = figure(figsize=(15.75, 10))
ax = fig.add_subplot(1,1,1) # one row, one column, first plot
ax.scatter(theta_data_l,theta_data_u, c=color_list)
ax.set_xlabel(lbl_lowerscale)
ax.set_ylabel(lbl_upperscale)
savefig('lowerscale_upperscale.png',dpi=120)