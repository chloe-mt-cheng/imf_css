import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from read_alf import Alf


font = {'family':'serif',
        'weight':'normal',
        'size':20
}
plt.rc('font',**font)

class alf_processing(Alf):
    """Class to help with processing alf fit files, making use of read_alf.py.
    """
        
    def __init__(self, path, wave_ranges, filetype):
        self.path = path
        self.wave_ranges = wave_ranges
        if filetype == 53:
            self.labels = ['velz', 'sigma', 'logage', 'zH', 'FeH', 'aH', 'CH', 'NH', 'NaH', 'MgH', 'SiH', 'KH', 'CaH', 'TiH', 'VH',
             'CrH', 'MnH', 'CoH', 'NiH', 'CuH', 'SrH', 'BaH', 'EuH', 'MgFe', 'Teff', 'IMF1', 'IMF2', 'logfy',
              'sigma2', 'velz2', 'hotteff', 'logm7g', 'loghot', 'fy_logage', 'logemline_h', 'logemline_oii', 
              'logemline_oiii', 'logemline_sii', 'logemline_ni', 'logemline_nii', 'logtrans', 'jitter', 'logsky', 
             'IMF3', 'IMF4', 'h3', 'h4', 'ML_v', 'MW_v', 'alpha', 'SiFe', 'CaFe', 'TiFe', 'ML_k']
        elif filetype == 50:
            self.labels = ['velz', 'sigma', 'logage', 'zH', 'FeH', 'aH', 'CH', 'NH', 'NaH', 'MgH', 'SiH', 'KH', 'CaH', 'TiH', 'VH',
             'CrH', 'MnH', 'CoH', 'NiH', 'CuH', 'SrH', 'BaH', 'EuH', 'MgFe', 'Teff', 'IMF1', 'IMF2', 'logfy',
              'sigma2', 'velz2', 'hotteff', 'logm7g', 'loghot', 'fy_logage', 'logemline_h', 
              'logemline_oiii', 'logemline_sii', 'logemline_ni', 'logemline_nii', 'logtrans', 'jitter', 'logsky', 
             'IMF3', 'IMF4', 'ML_v', 'MW_v', 'alpha', 'SiFe', 'CaFe', 'TiFe', 'ML_k']
        
        self.alf_result = None
        self.waves = None
        self.data_flux = None
        self.model_flux = None
        self.residuals = None
        self.unc = None
        self.ext_model_wave = None
        self.ext_model_flux = None
        self.lo_err = np.zeros(len(self.labels))
        self.hi_err = np.zeros(len(self.labels))
        self.fit_params = None
        
    def get_correct_alf_data(self, filetype):
        """Return the normalized and abundance-corrected alf fit results.

        Parameters
        ----------
        path : str
            Path to alf fit result files for a particular object (no extension)

        Returns
        -------
        alf_result : read_alf.Alf object
            alf fit results, normalized and abundance-corrected
        """

        self.alf_result = Alf(self.path, read_mcmc=True)
        self.alf_result.normalize_spectra()
        self.alf_result.get_total_met()
        self.alf_result.abundance_correct(m11=True)
        if filetype == 50:
            self.alf_result.model_info['Nsample'] = 1
        return self.alf_result

    def data_split(self, wave_range, alf_result):
        """Return the alf fit data, split by wavelength chunks.

        Parameters
        ----------
        wave_range : list
            Start and end of wavelength chunk
        alf_result : read_alf.Alf object
            alf fit results, normalized and abundance-corrected 

        Returns
        -------
        waves : tuple
            Wavelengths in wavelength chunk
        data_flux : tuple
            Flux of data in wavelength chunk
        model_flux : tuple
            Flux of model (fit) in wavelength chunk
        residuals : tuple
            Residuals between data and model in wavelength chunk
        unc : tuple
            Spectral uncertainty in wavelength chunk
        """

        self.waves = alf_result.spectra['wave'][(alf_result.spectra['wave'] > wave_range[0]) & (alf_result.spectra['wave'] \
                                           < wave_range[-1])]
        self.data_flux = alf_result.spectra['d_flux_norm'][(alf_result.spectra['wave'] > wave_range[0]) & \
                                                      (alf_result.spectra['wave'] < wave_range[-1])]
        self.model_flux = alf_result.spectra['m_flux_norm'][(alf_result.spectra['wave'] > wave_range[0]) & \
                                                      (alf_result.spectra['wave'] < wave_range[-1])]
        self.residuals = alf_result.spectra['residual'][(alf_result.spectra['wave'] > wave_range[0]) & \
                                                   (alf_result.spectra['wave'] < wave_range[-1])]
        self.unc = alf_result.spectra['unc'][(alf_result.spectra['wave'] > wave_range[0]) &  (alf_result.spectra['wave'] \
                                        < wave_range[-1])]
        if self.alf_result.ext_model != None:
            self.ext_model_wave = alf_result.ext_model['wave'][(alf_result.ext_model['wave'] > wave_range[0]) &  \
                                                               (alf_result.ext_model['wave'] < wave_range[-1])]
            self.ext_model_flux = alf_result.ext_model['flux'][(alf_result.ext_model['wave'] > wave_range[0]) &  \
                                                           (alf_result.ext_model['wave'] < wave_range[-1])]
        return self.waves, self.data_flux, self.model_flux, self.residuals, self.unc, self.ext_model_wave, self.ext_model_flux

    def alf_preprocessing(self, filetype):
        """Pre-process the alf fit results.  Essentially just calls the two functions above.

        Parameters
        ----------
        path : str
            Path to alf fit result files for a particular object (no extension)
        wave_ranges : tuple
            List of list of wavelength chunks

        Returns
        -------
        alf_result : read_alf.Alf object
            alf fit results, normalized and abundance-corrected 
        waves : tuple
            Wavelengths in all wavelength chunks
        data_flux : tuple
            Flux of data in all wavelength chunks
        model_flux : tuple
            Flux of model (fit) in all wavelength chunks
        residuals : tuple
            Residuals between data and model in all wavelength chunks
        unc : tuple
            Spectral uncertainty in all wavelength chunks
        """

        #Read alf fit results
        self.alf_result = alf_processing.get_correct_alf_data(self, filetype)

        #Split the data into wavelength chunks
        split_data = []
        for i in range(len(self.wave_ranges)):
            split_data.append(alf_processing.data_split(self, self.wave_ranges[i], self.alf_result))
        split_data = np.array(split_data, dtype=object)

        self.waves = split_data[:,0]
        self.data_flux = split_data[:,1]
        self.model_flux = split_data[:,2]
        self.residuals = split_data[:,3]
        self.unc = split_data[:,4]
        if self.alf_result.ext_model != None:
            self.ext_model_wave = split_data[:,5]
            self.ext_model_flux = split_data[:,6]
        
    def plot_3chunks(self, title_str):
        """Plot blue side only fits.
        
        Parameters
        ----------
        title_str : str
            Title for plot
        
        Returns
        -------
        None
        """
        
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 5), gridspec_kw={'height_ratios': [3, 1]})
        fig.set_tight_layout(True)
        fig.suptitle(title_str, fontsize=20, y=1.01)
        for i in range(len(self.waves)):
            ax[0,i].plot(self.waves[i], self.data_flux[i], color='dimgrey', lw=3)
            ax[0,i].plot(self.waves[i], self.model_flux[i], color='dodgerblue', alpha=0.7, lw=3)

            ax[1,i].plot(self.waves[i], self.residuals[i], color='dodgerblue', lw=2, alpha=0.7)
            ax[1,i].fill_between(self.waves[i], -(self.unc[i])*1e2, +(self.unc[i])*1e2, color='grey')

            ax[0,i].tick_params(length=6, width=2, labelsize=15)
            ax[1,i].tick_params(length=6, width=2, labelsize=15)
            ax[1,i].set_xlabel('Wavelength ($\AA$)', fontsize=15)
        ax[0,0].set_ylabel('Normalized Flux \n(Arb.)', fontsize=15)
        ax[1,0].set_ylabel('Residuals \n(%)', fontsize=15)
        
    def plot_5chunks(self, title_str):
        """Plot full wavelength range fits.
        
        Parameters
        ----------
        title_str : str
            Title for plot
        
        Returns
        -------
        None
        """
        
        fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(20, 10), gridspec_kw={'height_ratios':[3,1,3,1]})
        fig.set_tight_layout(True)
        fig.suptitle(title_str, fontsize=20, y=1.01)
        for i in range(0, 3):
            ax[0,i].plot(self.waves[i], self.data_flux[i], color='dimgrey', lw=3)
            ax[0,i].plot(self.waves[i], self.model_flux[i], color='r', alpha=0.7, lw=3)

            ax[1,i].plot(self.waves[i], self.residuals[i], color='r', lw=2, alpha=0.7)
            ax[1,i].fill_between(self.waves[i], -(self.unc[i])*1e2, +(self.unc[i])*1e2, color='grey')
            
            ax[0,i].tick_params(length=6, width=2, labelsize=15)
            ax[1,i].tick_params(length=6, width=2, labelsize=15)
        
        for i in range(0, 2):
            ax[2,i].plot(self.waves[i+3], self.data_flux[i+3], color='dimgrey', lw=3)
            ax[2,i].plot(self.waves[i+3], self.model_flux[i+3], color='r', alpha=0.7, lw=3)

            ax[3,i].plot(self.waves[i+3], self.residuals[i+3], color='r', lw=2, alpha=0.7)
            ax[3,i].fill_between(self.waves[i+3], -(self.unc[i+3])*1e2, +(self.unc[i+3])*1e2, color='grey')
            
            ax[2,i].tick_params(length=6, width=2, labelsize=15)
            ax[3,i].tick_params(length=6, width=2, labelsize=15)
            
        ax[2,2].axis('off')
        ax[3,2].axis('off')
        ax[0,0].set_ylabel('Normalized Flux \n(Arb.)', fontsize=15)
        ax[1,0].set_ylabel('Residuals \n(%)', fontsize=15)
        ax[2,0].set_ylabel('Normalized Flux \n(Arb.)', fontsize=15)
        ax[3,0].set_ylabel('Residuals \n(%)', fontsize=15)

    def calculate_param_errs(self):
        """Add upper and lower uncertainties of fitted/calculated parameters to object.
        """
        for i in range(len(self.labels)):
            if self.labels[i] == 'FeH':
                self.lo_err[i] = self.alf_result.tmet['cl50'] - self.alf_result.tmet['cl16']
                self.hi_err[i] = self.alf_result.tmet['cl84'] - self.alf_result.tmet['cl50']
            elif 'H' in self.labels[i] and 'z' not in self.labels[i] and 'Fe' not in self.labels[i]:
                self.lo_err[i] = self.alf_result.xH[self.labels[i][:-1]][5] - self.alf_result.xH[self.labels[i][:-1]][4]
                self.hi_err[i] = self.alf_result.xH[self.labels[i][:-1]][6] - self.alf_result.xH[self.labels[i][:-1]][5]
            elif self.labels[i][2:] == 'Fe':
                self.lo_err[i] = self.alf_result.xFe[self.labels[i][:-2]]['cl50'] - self.alf_result.xFe[self.labels[i][:-2]]['cl16']
                self.hi_err[i] = self.alf_result.xFe[self.labels[i][:-2]]['cl84'] - self.alf_result.xFe[self.labels[i][:-2]]['cl50']
            elif self.labels[i] == 'alpha':
                ML_v = np.where(self.alf_result.labels == 'ML_v')
                MW_v = np.where(self.alf_result.labels == 'MW_v')
                alpha_IMF_chain = np.sort(np.squeeze(self.alf_result.mcmc[:, ML_v]/self.alf_result.mcmc[:, MW_v]))
                num = self.alf_result.model_info['Nwalkers']*self.alf_result.model_info['Nchain']\
                /self.alf_result.model_info['Nsample']
                lower = alpha_IMF_chain[int(0.160*num)]
                median = alpha_IMF_chain[int(0.500*num)]
                upper = alpha_IMF_chain[int(0.840*num)]
                cl95 = alpha_IMF_chain[int(0.950*num)]
                std = np.std(alpha_IMF_chain)
                self.lo_err[i] = median - lower
                self.hi_err[i] = upper - median
            else:
                self.lo_err[i] = self.alf_result.results[self.labels[i]][5] - self.alf_result.results[self.labels[i]][4]
                self.hi_err[i] = self.alf_result.results[self.labels[i]][6] - self.alf_result.results[self.labels[i]][5]

    def get_fitted_params(self):
        """Add all fitted parameters and uncertainties to object.
        """
        
        inds = ['median', 'lo_err', 'hi_err']
        
        alf_processing.calculate_param_errs(self)

        col_vals = []
        for i in range(len(self.labels)):
            if self.labels[i] == 'FeH':
                col_vals.append([self.alf_result.tmet['cl50'], self.lo_err[i], self.hi_err[i]])
            elif 'H' in self.labels[i] and 'z' not in self.labels[i] and 'Fe' not in self.labels[i]:
                col_vals.append([self.alf_result.xH[self.labels[i][:-1]][5], self.lo_err[i], self.hi_err[i]])
            elif self.labels[i][2:] == 'Fe':
                col_vals.append([self.alf_result.xFe[self.labels[i][:-2]]['cl50'], self.lo_err[i], self.hi_err[i]])
            elif self.labels[i] == 'alpha':
                ML_v = np.where(self.alf_result.labels == 'ML_v')
                MW_v = np.where(self.alf_result.labels == 'MW_v')
                alpha_IMF_chain = np.sort(np.squeeze(self.alf_result.mcmc[:, ML_v]/self.alf_result.mcmc[:, MW_v]))
                num = self.alf_result.model_info['Nwalkers']*self.alf_result.model_info['Nchain']\
                /self.alf_result.model_info['Nsample']
                median = alpha_IMF_chain[int(0.500*num)]
                col_vals.append([median, self.lo_err[i], self.hi_err[i]])
            else:
                col_vals.append([self.alf_result.results[self.labels[i]][5], self.lo_err[i], self.hi_err[i]])
        col_vals = np.array(col_vals).T

        self.fit_params = pd.DataFrame(data=col_vals, columns=self.labels, index=inds)
        
    def fig3_plot(objects, obj_names, vel_disps, title_str, savestr):
        """Plot analogous to Fig. 3 in Villaume+2017.
        
        Parameters
        ----------
        objects : list
            List of alf results
        obj_names : list
            List of strings of object names
        vel_disps : list
            List of velocity dispersions
        title_str : str
            Title for plot
        savestr : str
            Filename under which to save plot
            
        Returns
        -------
        None
        """
        
        cmap = plt.cm.viridis
        nums = np.linspace(0, 1, len(objects))
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)
        fig.set_tight_layout(True)
        fig.suptitle(title_str, fontsize=20, y=1.01)

        for i in range(len(objects)):
            ax[0].errorbar(objects[i].fit_params['FeH']['median'], objects[i].fit_params['alpha']['median'], 
                           xerr=[[objects[i].fit_params['FeH']['lo_err']], [objects[i].fit_params['FeH']['hi_err']]],
                          yerr=[[objects[i].fit_params['alpha']['lo_err']], [objects[i].fit_params['alpha']['hi_err']]], 
                          linestyle='None', marker='.', markersize=20, color=cmap(nums[i]), label=obj_names[i])
            ax[1].errorbar(objects[i].fit_params['MgFe']['median'], objects[i].fit_params['alpha']['median'], 
                           xerr=[[objects[i].fit_params['MgFe']['lo_err']], [objects[i].fit_params['MgFe']['hi_err']]],
                          yerr=[[objects[i].fit_params['alpha']['lo_err']], [objects[i].fit_params['alpha']['hi_err']]], 
                          linestyle='None', marker='.', markersize=20, color=cmap(nums[i]))
            ax[2].errorbar(vel_disps[i], objects[i].fit_params['alpha']['median'], 
                           yerr=[[objects[i].fit_params['alpha']['lo_err']], [objects[i].fit_params['alpha']['hi_err']]], 
                           linestyle='None', marker='.', markersize=20, color=cmap(nums[i]))
        for i in range(len(ax)):
            ax[i].axhline(1, linestyle='dashed', color='k')
            ax[i].tick_params(length=6, width=2, labelsize=12)

        ax[0].set_xlabel('[Fe/H]', fontsize=15)
        ax[0].set_ylabel('$ɑ_{IMF} = (M/L)/(M/L)_{MW}$', fontsize=15)
        ax[0].legend(fontsize=12, loc='lower right', bbox_to_anchor= (-0.2,0))

        ax[1].set_xlabel('[Mg/Fe]', fontsize=15)

        ax[2].set_xlabel('$\sigma$', fontsize=15)
        plt.savefig(savestr, bbox_inches='tight')