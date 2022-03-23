import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii
import astropy.units as u
from astropy.nddata import StdDevUncertainty, InverseVariance
from astropy.modeling import models
from sedpy.observate import air2vac
from prospect import utils
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pandas as pd
import glob
from specutils.analysis import centroid
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler
from specutils.manipulation import extract_region
from specutils.fitting import fit_lines
import matplotlib.pyplot as plt
import flexure_correction as f

font = {'family':'serif',
        'weight':'normal',
        'size':20
}
plt.rc('font',**font)

#############
# Blue side #
#############

#Test with B058
flex_path = '../B058_old_blue_coadd1d.fits'
template_path = '../mock_b058_old_template.dat'

#Test finding redshift of entire spectrum
test_z, test_chi = test_pre_full(flex_path, template_path, -0.000741, 0.005, 'coadd')
print(test_z, test_chi)

#Measure redshift in chunks
noflex_path = '../B058_old_blue_coadd1d.fits'
data_wave_full, data_cut_wave, data_flux_full, data_cut_flux, data_noise_full, data_cut_noise = prep_data(noflex_path, 
                                                                                                         'coadd')
norm_blue_wave_chunks, norm_blue_flux_chunks, norm_blue_noise_chunks = data_chunks(data_cut_wave, data_cut_flux, data_cut_noise,
                                                                                   250)
norm_template_wave_chunks, norm_template_flux_chunks, norm_template_noise_chunks, template_central_wavelengths, \
central_wavelengths = template_chunks(data_cut_wave, data_cut_flux, data_cut_noise, template_path, -0.000741, 250, 20, 'before')
QA_chunks(norm_blue_wave_chunks, norm_blue_flux_chunks, norm_template_wave_chunks, norm_template_flux_chunks)
bestfit_redshift, best_chi2, redshifted_spectra, chi2 = chunk_redshift(data_cut_wave, data_cut_flux, data_cut_noise,
                                                                       template_path, -0.000741, 250, 20, -0.001828, 0.005, 
                                                                       'after')
redshifted_waves = []
redshifted_fluxes = []
for i in range(len(redshifted_spectra)):
    redshifted_waves.append(redshifted_spectra[i].spectral_axis)
    redshifted_fluxes.append(redshifted_spectra[i].flux)
QA_chunks(norm_blue_wave_chunks, norm_blue_flux_chunks, redshifted_waves, redshifted_fluxes)
QA_fit(bestfit_redshift[:-1], template_central_wavelengths[:-1])
fit, fit_z, fit_kms, error_func, error_func_z, error_func_kms, errors_A = fit_errors(bestfit_redshift[:-1], 
                                                                                     template_central_wavelengths[:-1])
corr_factor, wave_corr = flexure_function(fit, -0.000741, data_cut_wave)
flexure_QA(data_cut_wave, wave_corr)

#Re-do redshifting to make sure correction worked
tm_result_corr, dered_wave = dered_corr(noflex_path, template_path, wave_corr, -0.000741, 0.005, 'coadd')
print(tm_result_corr[0])
norm_blue_wave_chunks_corr, norm_blue_flux_chunks_corr, norm_blue_noise_chunks_corr = data_chunks(dered_wave, data_cut_flux, 
                                                                                                  data_cut_noise,
                                                                                   250)
norm_template_wave_chunks_corr, norm_template_flux_chunks_corr, norm_template_noise_chunks_corr, \
template_central_wavelengths_corr, central_wavelengths_corr = template_chunks(dered_wave, data_cut_flux, data_cut_noise, 
                                                                         template_path, -0.000741, 250, 20, 'after')
QA_chunks(norm_blue_wave_chunks_corr, norm_blue_flux_chunks_corr, norm_template_wave_chunks_corr, 
          norm_template_flux_chunks_corr)
bestfit_redshift_corr, best_chi2_corr, redshifted_spectra_corr, chi2_corr = chunk_redshift(dered_wave, data_cut_flux, 
                                                                                           data_cut_noise, template_path, 
                                                                                           -0.000741, 250, 20, -0.001828, 0.005,
                                                                                          'after')
redshifted_waves_corr = []
redshifted_fluxes_corr = []
for i in range(len(redshifted_spectra_corr)):
    redshifted_waves_corr.append(redshifted_spectra_corr[i].spectral_axis)
    redshifted_fluxes_corr.append(redshifted_spectra_corr[i].flux)
QA_chunks(norm_blue_wave_chunks_corr, norm_blue_flux_chunks_corr, redshifted_waves_corr, redshifted_fluxes_corr)
QA_fit(bestfit_redshift_corr[:-1], central_wavelengths_corr[:-1])

#Final corrected wavelength
corr_factor, blue_wave_corr = flexure_function(fit, -0.000741, data_wave_full)

############
# Red side #
############

#Test with B058
B058_old_red_file = glob.glob('../spec1d*B058*.fits')[0]
sky_paranal_filename = '/Applications/anaconda3/lib/python3.8/site-packages/pypeit/data/sky_spec/paranal_sky.fits'

waves_1d_red, fluxes_1d_red = read_1dspec(B058_old_red_file)
sky_paranal_wave, sky_paranal_flux = read_skymod(sky_paranal_filename)

#Smooth the sky model
R_red = data_resolution(sky_paranal_wave, sky_paranal_flux)
smoothed_sky_flux, smoothed_sky_noise = smooth_red(sky_paranal_wave, sky_paranal_flux)

#Find sky centroids 
sky_cents = sky_centroids(sky_paranal_wave, smoothed_sky_flux, smoothed_sky_noise)

#Get offsets between data and model
red_targ_bounds = [[7835, 7850], [7910, 7922], [7990, 8000], [8340, 8353], [8460, 8475], [8823, 8833], [8918, 8925], 
                   [9370, 9385], [9435, 9447], [9790, 9798]]
nearest_waves, offsets = centroid_offsets(red_targ_bounds, waves_1d_red, fluxes_1d_red, sky_cents)

#Fit the errors
fit, fit_kms, fit_func, fit_func_kms = offset_fit(nearest_waves, offsets)
QA_offset_fit(nearest_waves, offsets)

#Import co-added spectra
B058_old_red_coadd = fits.open('../B058_old_red_coadd1d.fits')
B058_old_red_coadd_wave = B058_old_red_coadd[1].data['wave']
B058_old_red_coadd_flux = B058_old_red_coadd[1].data['flux']
B058_old_red_coadd_noise = np.sqrt(B058_old_red_coadd[1].data['ivar'])
B058_old_red_coadd.close()

#Correct the wavelength
wave_corr = fit_interp(B058_old_red_coadd_wave, B058_old_red_coadd_flux, B058_old_red_coadd_noise, nearest_waves, fit_func)

#Re-do to check if correction worked
waves_1d_red_corr = fit_interp(waves_1d_red, fluxes_1d_red, np.zeros_like(fluxes_1d_red), nearest_waves, fit_func)

red_targ_bounds_corr = [[7835, 7850], [7910, 7922], [7990, 8000], [8340, 8353], [8460, 8475], [8824, 8835], [8918, 8925], 
                   [9370, 9385], [9437, 9448], [9790, 9798]]
nearest_waves_corr, offsets_corr = centroid_offsets(red_targ_bounds_corr, waves_1d_red_corr, fluxes_1d_red, sky_cents)

fit_corr, fit_corr_kms, fit_func_corr, fit_func_corr_kms = offset_fit(nearest_waves_corr, offsets_corr)
QA_offset_fit(nearest_waves_corr, offsets_corr)