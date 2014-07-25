import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from bb1truncpl import BB1TruncPL

class SimLFUtil:
    """
    This class is a helper class for generating flux data.
    
    It contains functions which are useful for this purpose:
        - generating luminosity samples and random distances for flux data,
        - determining fluxes with Gaussian noise based on foregoing data,
        - filtering most noisy data with threshold,
        - plotting figures of simulations,
        - writing fluxes and related distances to file

    The above-mentioned functions can be used separately or can be invoked
    by a batch function.
    """
    
    # The directory of the results:
    data_dir = 'c:/temp/data/'

    def __init__(self, args):
        self.gamma = args.gamma
        self.lower_scale = args.lower_scale
        self.scale = args.scale
        self.n_samp = args.n_samp
        self.sdev = args.sdev
        self.r_max = args.r_max
        self.thr_coef = args.thr_coef
        self.n_bins = args.n_bins
        self.bb1 = BB1TruncPL(args.gamma, args.lower_scale, args.scale)
        self.mean = 0.0

    def lum_sample(self):
        """
        This function returns generated luminosity samples based on BB1TruncPL
        """
        l_array, i = self.bb1.sample_array(self.n_samp,1)
        l_array = l_array[:self.n_samp]
        return l_array

    def rnd_distance_sample(self):
        """
        This function returns random distances for flux data (max. self.r_max Mpc)
        where CDF of distances: F(x) = (x**3)/(r_max**3)
        """
        u_array = np.random.uniform(0.0,1.0,self.n_samp)
        # applied inverse function of CDF of distances:
        r_array = np.multiply(self.r_max, np.power(u_array,1.0/3.0))
        return r_array

    def flux_sample(self, lums, dists):
        """
        This function returns flux samples based on input luminosity data (lums)
        and random distances (dists) arguments.
        """
        f_array = np.divide(lums,np.multiply(4*np.pi,np.power(dists,2)))
        return f_array

    def filtered_noisy_flux_sample(self, fluxes):
        """
        This function returns flux data with Gaussian noise (with mean = 0,
        standard deviation = self.sdev) where the most noisy data is filtered by
        self.thr_coef * self.sdev threshold.
        """
        g_array = np.random.normal(self.mean,self.sdev,self.n_samp)
        thres = self.thr_coef * self.sdev
        mask = np.absolute(g_array) < thres
        noisy = fluxes + g_array
        # Masking:
        filtered_noisy = noisy[mask]
        return filtered_noisy

    def save_data_to_disk(self,dists,fluxes,noisy_fluxes):
        """
        This function saves the random distances, flux data
        and the related filtered noisy data to disk.
        """
        np.savetxt(self.data_dir + 'dists' + '_cnt_' + str(self.n_samp) + '.dat', dists, fmt='%10.6e')
        np.savetxt(self.data_dir + 'fluxes' + '_cnt_' + str(self.n_samp) + '.dat', fluxes, fmt='%10.6e')
        n_fluxes_len = len(noisy_fluxes)
        noisy_fluxes_w_sdev = noisy_fluxes
        noisy_fluxes_w_sdev.shape = (1,n_fluxes_len)
        sdev_array = np.linspace(self.sdev,self.sdev,n_fluxes_len)
        sdev_array.shape = (1,n_fluxes_len)
        noisy_fluxes_w_sdev = np.concatenate((noisy_fluxes_w_sdev.T,sdev_array.T),axis=1)
        np.savetxt(self.data_dir + 'filtered_fluxes_w_G_noise_mu_' + str(self.mean) + '_sig_' + str(self.sdev) + '_cnt_' + str(self.n_samp) + '.dat', noisy_fluxes_w_sdev, fmt='%10.6e')
        
    def _save_distance_hists_plot(self,dists,n_bins):
        """
        This function saves the histograms of random distances.
        """
        plt.figure()
        plt.hist(dists, bins=np.linspace(dists.min(),dists.max(),n_bins+1), label='distances')
        plt.legend(loc='upper right')
        plt.savefig(self.data_dir + 'fig_distances_hist.png',dpi=120)

    def _save_flux_hists_plot(self,fluxes,noisy_fluxes,n_bins):
        """
        This function saves the histograms of flux data and the related filtered noisy data.
        """
        m_fluxes = fluxes[fluxes>0]
        m_noisy = noisy_fluxes[noisy_fluxes>0]
        lbins_fluxes = np.logspace(np.log10(m_fluxes.min()),np.log10(m_fluxes.max()),n_bins+1)
        lbins_noisy = np.logspace(np.log10(m_noisy.min()),np.log10(m_noisy.max()),n_bins+1)
        plt.figure()
        plt.hist(m_fluxes, bins=lbins_fluxes, label='flux', log=True)
        plt.hist(m_noisy, bins=lbins_noisy, label='noisy flux filtered by thres', log=True)
        plt.legend(loc='upper right')
        plt.savefig(self.data_dir + 'fig_flux_hists.png',dpi=120)

    def _sample_stats(self,smpl):
        """
        This function prints the basic statistics of input smpl.
        """
        print('Min of samples:',smpl.min())
        print('Max of samples:',smpl.max())
        print('Range of samples:',smpl.max()-smpl.min())
        print('Mean of samples:',smpl.mean())

    def batch_process(self):
        """
        This function contains invoking of all utility function.
        """
        t0 = dt.datetime.today()
        lums = self.lum_sample()
        t1 = dt.datetime.today()
        print('Elapsed time of generating luminosity sample:', t1-t0)
        
        t0 = dt.datetime.today()
        dists = self.rnd_distance_sample()
        t1 = dt.datetime.today()
        print('Elapsed time of generating random distances:', t1-t0)
        
        t0 = dt.datetime.today()
        fluxes = self.flux_sample(lums,dists)
        t1 = dt.datetime.today()
        print('Elapsed time of determining fluxes:', t1-t0)
        
        t0 = dt.datetime.today()
        noisy_fluxes = self.filtered_noisy_flux_sample(fluxes)
        t1 = dt.datetime.today()
        print('Elapsed time of determining filtered noisy fluxes:', t1-t0)
        
        t0 = dt.datetime.today()
        self._save_flux_hists_plot(fluxes,noisy_fluxes,self.n_bins)
        t1 = dt.datetime.today()
        print('Elapsed time of plotting:', t1-t0)
        
        t0 = dt.datetime.today()
        print('Basic statistics for:')
        print('----------------')
        print('Random distances')
        self._sample_stats(dists)
        print('------')
        print('Fluxes')
        self._sample_stats(fluxes)
        print('---------------------')
        print('Filtered noisy fluxes')
        self._sample_stats(noisy_fluxes)
        t1 = dt.datetime.today()
        print('Elapsed time of calculating statistics:', t1-t0)
        
        t0 = dt.datetime.today()
        self.save_data_to_disk(dists,fluxes,noisy_fluxes)
        t1 = dt.datetime.today()
        print('Elapsed time of data files writing:', t1-t0)