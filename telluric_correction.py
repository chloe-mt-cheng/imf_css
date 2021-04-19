import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
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

#Alexa's wavelength-dependent resolution function
def lris_res(lb, lr):
    """Get wavelength dependent resolution.
    
    Parameters
    ----------
    lb, lr : Arrays of blue and red wavelengths in angstroms in vacuum.
    
    Returns
    -------
    resb, resr : Resolution in km/s
    """
    
    resb = 3.09 + 0.187*(lb - 5000.)/1000. #res in A
    resb = resb/lb*3.e5 #res in km/s
    l9 = (lr - 9000.)/1000.
    resr = 1.26 - 0.128*l9 + 0.168*l9**2 + 0.1173*l9**3
    resr = resr/lr*3.E5
    
    return resb, resr
    
    
def normalize(wavelength, flux):
    """Return a normalized spectrum using a 4th-order polynomial (used old continuum-normalization code
    as a model).
    
    Parameters
    ----------
    wavelength : tuple
        Wavelength array
    flux : tuple
        Flux array
        
    Returns
    -------
    flux_norm : tuple
        Normalized flux
    """
    
    #Fit a 4th-order polynomial to the spectrum
    lambda_min = np.min(wavelength)
    lambda_max = np.min(wavelength)
    xrange = np.linspace(np.min(lambda_min), np.max(lambda_max), len(wavelength))
    polynomial_fit = np.polyfit(xrange, flux, 4)
    poly_obj = np.poly1d(polynomial_fit)
    
    #Divide out the fitted polynomial to get a continuum-normalized spectrum
    flux_norm = flux/poly_obj(xrange)
    
    return flux_norm
    
    
def minimization(f, correction_template_flux, correction_gal_flux):
	"""Minimization for scipy.optimize.minimize to find the scale factor between the template and target spectra.
	
	Parameters
	----------
	f : float
		Initial guess for scale factor
	correction_template_flux : tuple
		Flux array over minimization range for the template
	correction_gal_flux : tuple
		Flux array over minimization range for target
		
	Returns
	-------
	np.max(np.abs(correction_gal_flux - Tf)) : float
		The maximum absolute difference between the target flux and scaled template flux
	"""
	
	#Scale the template flux 
    Tf = 1 + f*(correction_template_flux - 1)
    
    #Minimize
    return np.max(np.abs(correction_gal_flux - Tf))
    
    
def Tf(f, T0):
    """Return the scaled template spectrum.
    
    Parameters
    ----------
    f : float
        Scale parameter
    T0 : tuple
        Template atmospheric transmission spectrum
    
    Returns
    -------
    1 + f*(T0 - 1) : tuple
        Scaled template spectrum
    """
    
    return 1 + f*(T0 - 1)


def telluric_corrections(blue_path, red_path):
	"""Return the tellurically-corrected flux and corresponding wavelength for a reduced object.
	
	Parameters
	----------
	blue_path : str
		Path to the blue coadded spectrum
	red_path : str
		Path to the red coadded spectrum
		
	Returns
	-------
	total_target_wave : tuple
		Wavelength array
	telluric_corrected_flux : tuple
		Tellurically-corrected flux array
	"""
	
	#Import telluric grid
	file = '/Applications/anaconda3/lib/python3.8/site-packages/pypeit/data/telluric/TelFit_MaunaKea_3100_26100_R20000.fits'
	hdul = fits.open(file)
	header0 = hdul[0].header
	data0 = hdul[0].data #flux
	header1 = hdul[1].header
	data1 = hdul[1].data #wavelength
	hdul.close()
	
	#Import co-added target spectra
	blue_coadd1d = fits.open(blue_path)
	blue_dat = blue_coadd1d[1].data
	blue_coadd1d.close()
	
	red_coadd1d = fits.open(red_path)
	red_dat = red_coadd1d[1].data
	red_coadd1d.close()
	
	#Pick one atmospheric transmission template
	template_wave = data1*10 #Convert from nm to A
	template_flux = data0[3,1,1,1]
	
	#Get wavelengths and fluxes for target and template
	target_blue_wave = blue_dat['wave']
	target_red_wave = red_dat['wave']
	target_blue_flux = blue_dat['flux']
	target_red_flux = red_dat['flux']
	
	template_blue_wave = template_wave[(template_wave >= np.min(target_blue_wave)) & (template_wave <= np.max(target_blue_wave))]
	template_red_wave = template_wave[(template_wave >= np.min(target_red_wave)) & (template_wave <= np.max(target_red_wave))]
	template_blue_flux = template_flux[(template_wave >= np.min(target_blue_wave)) & (template_wave <= np.max(target_blue_wave))]
	template_red_flux = template_flux[(template_wave >= np.min(target_red_wave)) & (template_wave <= np.max(target_red_wave))]
	
	#Get resolution using wavelength ranges from target object
	resb_template, resr_template = lris_res(template_blue_wave, template_red_wave)
	
	#Concatenate into one wavelength range
	total_template_res = np.concatenate((resb_template, resr_template))
	total_template_wave = np.concatenate((template_blue_wave, template_red_wave))
	total_template_flux = np.concatenate((template_blue_flux, template_red_flux))
	total_target_wave = np.concatenate((target_blue_wave, target_red_wave))
	total_target_flux = np.concatenate((target_blue_flux, target_red_flux))
	
	#Convert telluric resolution to km/s
	c = 299792.458 #speed of light
	R = 20000 #instrument resolution
	inval = total_template_wave/R
	in_sigma = inval/2.*np.sqrt(2.*np.log(2))
	in_sigma_kms = (in_sigma/total_template_wave)*c
	
	#Smooth template spectrum
	sigma_aa_desired = total_template_res/c*total_template_wave
	sigma_aa_original = in_sigma_kms/c*total_template_wave
	delta_sigma_aa_vector = np.sqrt(sigma_aa_desired**2 - sigma_aa_original**2)
	smoothed_template_flux = utils.smoothing.smoothspec(total_template_wave, total_template_flux, outwave=total_template_wave, smoothtype='lsf', resolution=delta_sigma_aa_vector, fftsmooth=True)
	
	#Re-grid template spectrum by interpolating onto target wavelength grid
	regrid_template_flux = np.interp(total_target_wave, total_template_wave, smoothed_template_flux)
	
	#Normalize the target and template spectra over 9250-9650A
	normrange_wave = total_target_wave[(total_target_wave >= 9250) & (total_target_wave <= 9650)]
	normrange_target_flux = total_target_flux[(total_target_wave >= 9250) & (total_target_wave <= 9650)]
	normrange_template_flux = regrid_template_flux[(total_target_wave >= 9250) & (total_target_wave <= 9650)]
	normed_template_flux = normalize(normrange_wave, normrange_template_flux)
	normed_target_flux = normalize(normrange_wave, normrange_target_flux)
	
	#Do the minimization over 9320-9380A
	correction_wave = normrange_wave[(normrange_wave >= 9320) & (normrange_wave <= 9380)]
	correction_template_flux = normed_template_flux[(normrange_wave >= 9320) & (normrange_wave <= 9380)]
	correction_target_flux = normed_target_flux[(normrange_wave >= 9320) & (normrange_wave <= 9380)]
	
	x0 = 1 #Initial guess
	res = op.minimize(minimization, x0, args=(correction_template_flux, correction_target_flux), method='Nelder-Mead', tol=1e-6)
	
	#Scale the template spectrum by the best-fit scale factor 
	scaled_template_flux = Tf(res.x, regrid_template_flux)
	
	#Tellurically-correct the target spectrum
	telluric_corrected_flux = total_target_flux/scaled_template_flux
	
	return total_target_wave, telluric_corrected_flux