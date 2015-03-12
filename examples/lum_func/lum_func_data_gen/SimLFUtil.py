"""
Created 2014-07-21 by Janos Szalai-Gindl and Tamas Budavari;
"""
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
        - optional: filtering most noisy data with flux limit
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
        self.n_bins = args.n_bins
        self.bb1 = BB1TruncPL(args.gamma, args.lower_scale, args.scale)
        self.mean = 0.0
        self.flux_limit = args.flux_limit

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

    def noisy_flux_sample(self, fluxes):
        """
        This function returns flux data with Gaussian noise (with mean = 0,
        standard deviation = self.sdev)
        """
        noisy = []
        sdev_list = []		
        for fl in fluxes:
            actual_sdev = np.sqrt((self.sdev)**2 + (0.01*fl)**2)
            #print 'sdev: ',self.sdev, ' fl: ', fl, ' actual_sdev: ', actual_sdev
            sdev_list.append(actual_sdev)
            noise = np.random.normal(self.mean, actual_sdev)
            var = fl + noise
            noisy.append(var)
        noisy = np.array(noisy)
        sdev_list = np.array(sdev_list)
        return noisy, sdev_list

    def filtered_noisy_flux_samples_with_flux_limit(self):
        """
        This function returns flux data with Gaussian noise (with mean = 0,
        standard deviation = self.sdev) where the flux data is kept
        if it is over the flux limit.
        """
        noisy_fluxes_with_limit = np.array([],dtype=np.float64)
        sdev_list_out = np.array([],dtype=np.float64)
        lums_with_limit = np.array([],dtype=np.float64)
        dists_with_limit = np.array([],dtype=np.float64)
        fluxes_with_limit = np.array([],dtype=np.float64)
        iter = 0
        droppedNum = 0
        while noisy_fluxes_with_limit.size < self.n_samp:
            lums = self.lum_sample()
            dists = self.rnd_distance_sample()
            fluxes = self.flux_sample(lums,dists)
            noisy_fluxes, sdev_list = self.noisy_flux_sample(fluxes)
            mask = self.flux_limit < noisy_fluxes
            # Masking:
            t1 = noisy_fluxes[mask]
            droppedNum += (len(noisy_fluxes) - len(t1))
            noisy_fluxes_with_limit = np.hstack([noisy_fluxes_with_limit,t1])
            t2 = lums[mask]
            lums_with_limit = np.hstack([lums_with_limit,t2])
            t3 = dists[mask]
            dists_with_limit = np.hstack([dists_with_limit,t3])
            t4 = fluxes[mask]
            fluxes_with_limit = np.hstack([fluxes_with_limit,t4])
            t5 = sdev_list[mask]
            sdev_list_out = np.hstack([sdev_list_out,t5])
            iter += 1
        return noisy_fluxes_with_limit[:self.n_samp], sdev_list_out[:self.n_samp], lums_with_limit[:self.n_samp], dists_with_limit[:self.n_samp], fluxes_with_limit[:self.n_samp], iter, droppedNum

    def save_data_to_disk(self,lums,dists,fluxes,noisy_fluxes,sdev_list,with_limit=False):
        """
        This function saves the random distances, flux data
        and the related filtered noisy data to disk.
        """
        np.savetxt(self.data_dir + 'lums' + '_cnt_' + str(self.n_samp) + '.dat', lums, fmt='%10.6e')
        np.savetxt(self.data_dir + 'dists' + '_cnt_' + str(self.n_samp) + '.dat', dists, fmt='%10.6e')
        np.savetxt(self.data_dir + 'fluxes' + '_cnt_' + str(self.n_samp) + '.dat', fluxes, fmt='%10.6e')
        n_fluxes_len = len(noisy_fluxes)
        noisy_fluxes_w_sdev = noisy_fluxes
        noisy_fluxes_w_sdev.shape = (1,n_fluxes_len)
        print sdev_list.shape
        sdev_list.shape = (1,n_fluxes_len)
        noisy_fluxes_w_sdev = np.concatenate((noisy_fluxes_w_sdev.T,sdev_list.T),axis=1)
        if with_limit:
            np.savetxt(self.data_dir + 'filtered_fluxes_w_G_noise_mu_' + str(self.mean) + '_sig_' + str(self.sdev) + '_fl_' + str(self.flux_limit) + '_cnt_' + str(self.n_samp) + '.dat', noisy_fluxes_w_sdev, fmt='%10.6e')
        else:
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
        if self.flux_limit == 0.0:
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
            noisy_fluxes, sdev_list = self.noisy_flux_sample(fluxes)
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
            print('Noisy fluxes')
            self._sample_stats(noisy_fluxes)
            t1 = dt.datetime.today()
            print('Elapsed time of calculating statistics:', t1-t0)
            
            t0 = dt.datetime.today()
            self.save_data_to_disk(lums,dists,fluxes,noisy_fluxes,sdev_list)
            t1 = dt.datetime.today()
            print('Elapsed time of data files writing:', t1-t0)
        else:
            t0 = dt.datetime.today()
            noisy_fluxes_with_limit, sdev_list, lums_with_limit, dists_with_limit, fluxes_with_limit, iter, droppedNum = self.filtered_noisy_flux_samples_with_flux_limit()
            t1 = dt.datetime.today()
            print('Elapsed time of determining filtered noisy fluxes with flux limit:', t1-t0)
            print('Necessary iteration for it:', iter)
            print('Number of dropped flux samples:', droppedNum)
            
            t0 = dt.datetime.today()
            print('Basic statistics for filtered noisy fluxes with flux limit:')
            self._sample_stats(noisy_fluxes_with_limit)
            t1 = dt.datetime.today()
            print('Elapsed time of calculating statistics:', t1-t0)
            
            t0 = dt.datetime.today()
            self.save_data_to_disk(lums_with_limit,dists_with_limit,fluxes_with_limit,noisy_fluxes_with_limit,sdev_list, True)
            t1 = dt.datetime.today()
            print('Elapsed time of data files writing:', t1-t0)