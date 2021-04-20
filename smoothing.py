import telluric_correction as tell
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
from astropy.convolution import interpolate_replace_nans, Gaussian1DKernel, convolve, convolve_fft
import sys
import glob
from sedpy.observate import air2vac
from prospect import utils
from scipy.interpolate import interp1d
import scipy.optimize as op
import pandas as pd

font = {'family':'serif',
        'weight':'normal',
        'size':20
}
plt.rc('font',**font)

def spikes(wave, flux, hi, lo, mid):
	spikes_hi = np.argwhere(flux > hi)
	spikes_lo = np.argwhere(flux < lo)
	
	#Get mid-spectra spikes
	inds = []
	for i in range(len(flux)):
		if wave[i] >= 7500 and wave[i] <= 8000:
			inds.append(i)
			
	spikes_mid = []
	for i in range(len(flux[inds])):
		if flux[inds][i] <= mid:
			spikes_mid.append(inds[i])
			
	return spikes_hi, spikes_lo, spikes_mid
	

def smoothing(blue_path, red_path, tell_wave, tell_flux, hi, lo, mid):
	#Import co-added target spectra
	blue_coadd1d = fits.open(blue_path)
	blue_dat = blue_coadd1d[1].data
	blue_coadd1d.close()
	
	red_coadd1d = fits.open(red_path)
	red_dat = red_coadd1d[1].data
	red_coadd1d.close()
	
	#Get wavelengths and fluxes
	blue_wave = blue_dat['wave'][blue_dat['wave'] >= 4000]
	red_wave = red_dat['wave']
	blue_ivar = blue_dat['ivar'][blue_dat['wave'] >= 4000]
	red_ivar = red_dat['ivar']
	
	flux = tell_flux[tell_wave >= 4000]
	total_wave = np.concatenate((blue_wave, red_wave))
	noise = 1/np.sqrt(np.concatenate((blue_ivar, red_ivar)))
	
	#Get resolution
	resb, resr = tell.lris_res(blue_wave, red_wave)
	total_res = np.concatenate((resb, resr))
	
	#Get pixel positions of noisy telluric spikes
	spikes_hi, spikes_lo, spikes_mid = spikes(total_wave, flux, hi, lo, mid)
	bad_pixels = []
	for i in range(len(spikes_hi)):
		for j in range(len(spikes_hi[i])):
			bad_pixels.append(spikes_hi[i][j])
	for i in range(len(spikes_lo)):
		for j in range(len(spikes_lo[i])):
			bad_pixels.append(spikes_lo[i][j])
	for i in range(len(spikes_mid)):
		bad_pixels.append(spikes_mid[i])
		
	bad_pixels = np.sort(list(set(bad_pixels))) #Remove duplicates
	
	#Interpolate over noisy spikes
	flux[bad_pixels] = np.nan
	noise[bad_pixels] = np.nan
	
	kernel = Gaussian1DKernel(stddev = 1)
	intp_flux = interpolate_replace_nans(flux, kernel, convolve = convolve_fft)
	intp_noise = interpolate_replace_nans(noise, kernel, convolve = convolve_fft)
	
	#Smooth to max in 4000A+ range
	c = 299792.458 #speed of light
	in_sigma_kms = np.max(total_res) + 0.5
	sigma_aa_original = total_res/c*total_wave
	sigma_aa_desired = in_sigma_kms/c*total_wave
	delta_sigma_aa_vector = np.sqrt(sigma_aa_desired**2 - sigma_aa_original**2)
	smoothed_flux = utils.smoothing.smoothspec(total_wave, intp_flux, outwave=total_wave, smoothtype='lsf', resolution=delta_sigma_aa_vector)
	smoothed_noise = utils.smoothing.smoothspec(total_wave, intp_noise, outwave=total_wave, smoothtype='lsf', resolution=delta_sigma_aa_vector)
	return total_wave, smoothed_flux, smoothed_noise
		