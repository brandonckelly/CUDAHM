__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import os

# physical constants, cgs
clight = 2.99792458e10
hplanck = 6.6260755e-27
kboltz = 1.380658e-16

wavelength = np.asarray([100.0, 160.0, 250.0, 350.0, 500.0])  # observational wavelengths in microns
nu = clight / (wavelength / 1e4)
nu.sort()
print(nu)
nu_ref = 2.3e11  # 230 GHz


def modified_bbody(nu, const, beta, temp):
    sed = 2.0 * hplanck * nu ** 3 / (clight * clight) / (np.exp(hplanck * nu / (kboltz * temp)) - 1.0)
    sed *= const * (nu / nu_ref) ** beta
    return sed


ndata = 100000

cbt_mean = np.asarray([15.0, 2.0, np.log(15.0)])

cbt_sigma = np.asarray([1.0, 0.1, 0.3])
cbt_corr = np.asarray([[1.0, -0.5, 0.0],
                       [-0.5, 1.0, 0.25],
                       [0.0, 0.25, 1.0]])

cbt_cov = np.dot(np.diag(cbt_sigma), cbt_corr.dot(np.diag(cbt_sigma)))

cbt = np.random.multivariate_normal(cbt_mean, cbt_cov, ndata)

data_dir = os.environ['HOME'] + '/Projects/CUDAHM/dusthm/data/'
np.savetxt(data_dir + 'true_cbt_' + str(ndata) + '.dat', cbt, fmt='%10.6e')

sed = np.zeros((ndata, len(nu)))

for j in range(len(nu)):
    sed[:, j] = modified_bbody(nu[j], np.exp(cbt[:, 0]), cbt[:, 1], np.exp(cbt[:, 2]))

# generate noise assuming a median S/N of 200
fnu_sig = np.median(sed, axis=0) / 1000.0
fnu_sig = np.outer(np.ones(ndata), fnu_sig)

fnu = sed + fnu_sig * np.random.standard_normal(fnu_sig.shape)

data = np.zeros((ndata, 2*len(nu)))
data[:, 0] = fnu[:, 0]
data[:, 1] = fnu_sig[:, 0]
data[:, 2] = fnu[:, 1]
data[:, 3] = fnu_sig[:, 1]
data[:, 4] = fnu[:, 2]
data[:, 5] = fnu_sig[:, 2]
data[:, 6] = fnu[:, 3]
data[:, 7] = fnu_sig[:, 3]
data[:, 8] = fnu[:, 4]
data[:, 9] = fnu_sig[:, 4]

data_dir = os.environ['HOME'] + '/Projects/CUDAHM/dusthm/data/'
header = 'nu = '
for j in range(len(nu)):
    header += str(nu[j]) + ', '
np.savetxt(data_dir + 'cbt_sed_' + str(ndata) + '.dat', data, fmt='%10.6e', header=header)

idx = np.random.random_integers(0, fnu.shape[0]-1)
plt.errorbar(nu, fnu[idx], yerr=fnu_sig[idx])
plt.xscale('log')
nurange = nu.max() - nu.min()
plt.xlim(nu.min() - 0.05 * nurange, nu.max() + 0.05 * nurange)
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'$f_{\nu}$ [arbitrary]')
plt.title('Index ' + str(idx))
plt.show()