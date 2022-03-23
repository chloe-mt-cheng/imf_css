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

font = {'family':'serif',
        'weight':'normal',
        'size':20
}
plt.rc('font',**font)

#############################################################
# Continuum-normalization functions (also normalizes noise) #
#############################################################
def data_ranges(lambda_min, lambda_max, flux, wavelength, noise):
    """Return the data within the desired wavelength range.
    
    Parameters
    ----------
    lambda_min : float
        The lower bound of the wavelength range 
    lambda_max : float
        The upper bound of the wavelength range
    flux : tuple
        The flux data
    wavelength : tuple
        The wavelength data
    noise : tuple
        The noise data
        
    Returns
    -------
    wavelength_inrange : tuple
        The wavelength data within the desired wavelength range
    flux_inrange : tuple
        The flux data within the desired wavelength range
    noise_inrange : tuple
        The noise data within the desired wavelength range
    """
    
    rng = (wavelength >= lambda_min) & (wavelength <= lambda_max)
    wavelength_inrange = wavelength[rng]
    flux_inrange = flux[rng]
    noise_inrange = noise[rng]
    return wavelength_inrange, flux_inrange, noise_inrange

def poly_order(lambda_min, lambda_max):
    """Return the order of the polynomial by which to continuum-normalize the spectrum.
    
    Parameters
    ----------
    lambda_min : float
        The lower bound of the wavelength range
    lambda_max : float
        The upper bound of the wavelength range
        
    Returns
    -------
    (lambda_min/lambda_max)/100 : int
        The order of the polynomial by which to continuum-normalize the spectrum
    """
    
    return int((lambda_max - lambda_min)/100)

def continuum_normalize(lambda_min, lambda_max, flux, wavelength, noise):
    """Return the spectrum normalized by the fitted polynomial shape of the continuum.
    
    Parameters
    ----------
    lambda_min : float
        The lower bound of the wavelength range
    lambda_max : float
        The upper bound of the wavelength range
    flux : tuple
        The flux data
    wavelength: tuple
        The wavelength data
    noise : tuple
        The noise data
        
    Returns
    -------
    wavelength_inrange : tuple
        The wavelength data within the desired wavelength range
    flux_norm : tuple
        The continuum-normalized flux within the desired wavelength range
    noise_norm : tuple
        The continuum-normalized noise within the desired wavelength range
    """
    
    #Get the data in the desired wavelength range
    wavelength_inrange, flux_inrange, noise_inrange = data_ranges(lambda_min, lambda_max, flux, wavelength, noise)
    
    #Get the order of the polynomial 
    n = poly_order(lambda_min, lambda_max)
    
    #Fit an nth-order polynomial to the spectrum
    xrange = np.linspace(np.min(lambda_min), np.max(lambda_max), len(wavelength_inrange))
    polynomial_fit = np.polyfit(xrange, flux_inrange, n)
    poly_obj = np.poly1d(polynomial_fit)
    
    polynomial_fit_noise = np.polyfit(xrange, noise_inrange, n)
    poly_obj_noise = np.poly1d(polynomial_fit_noise)
    
    #Divide out the fitted polynomial to get a continuum-normalized spectrum
    flux_norm = flux_inrange/poly_obj(xrange)
    noise_norm = noise_inrange/poly_obj_noise(xrange)
    return wavelength_inrange, flux_norm, noise_norm

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
    
############################################
# Re-jigged specutils functions            #
# Note: PR on Mar. 22/22 to add my changes #
############################################

#Re-jigged specutils functions
def _uncertainty_to_standard_deviation(uncertainty):
    """
    Convenience function to convert other uncertainty types to standard deviation,
    for consistency in calculations elsewhere.

    Parameters
    ----------
    uncertainty : :class:`~astropy.nddata.NDUncertainty`
        The input uncertainty

    Returns
    -------
    :class:`~numpy.ndarray`
        The array of standard deviation values.

    """
    if uncertainty is not None:
        if isinstance(uncertainty, StdDevUncertainty):
            stddev = uncertainty.array
        elif isinstance(uncertainty, VarianceUncertainty):
            stddev = np.sqrt(uncertainty.array)
        elif isinstance(uncertainty, InverseVariance):
            stddev = 1 / np.sqrt(uncertainty.array)

        return stddev

def _resample(resample_method):
    """
    Find the user preferred method of resampling the template spectrum to fit
    the observed spectrum.

    Parameters
    ----------
    resample_method: `string`
        The type of resampling to be done on the template spectrum.

    Returns
    -------
    :class:`~specutils.ResamplerBase`
        This is the actual class that will handle the resampling.
    """
    if resample_method == "flux_conserving":
        return FluxConservingResampler()

    if resample_method == "linear_interpolated":
        return LinearInterpolatedResampler()

    if resample_method == "spline_interpolated":
        return SplineInterpolatedResampler()

    return None

def _normalize_for_template_matching(observed_spectrum, template_spectrum, stddev=None):
    """
    Calculate a scale factor to be applied to the template spectrum so the
    total flux in both spectra will be the same.

    Parameters
    ----------
    observed_spectrum : :class:`~specutils.Spectrum1D`
        The observed spectrum.
    template_spectrum : :class:`~specutils.Spectrum1D`
        The template spectrum, which needs to be normalized in order to be
        compared with the observed spectrum.

    Returns
    -------
    `float`
        A float which will normalize the template spectrum's flux so that it
        can be compared to the observed spectrum.
    """
    if stddev is None:
        stddev = _uncertainty_to_standard_deviation(observed_spectrum.uncertainty)
    #Changed to np.nansum
    num = np.nansum((observed_spectrum.flux*template_spectrum.flux) / (stddev**2))
    denom = np.nansum((template_spectrum.flux / stddev)**2)

    return num/denom

def _chi_square_for_templates(observed_spectrum, template_spectrum, resample_method):
    """
    Resample the template spectrum to match the wavelength of the observed
    spectrum. Then, calculate chi2 on the flux of the two spectra.

    Parameters
    ----------
    observed_spectrum : :class:`~specutils.Spectrum1D`
        The observed spectrum.
    template_spectrum : :class:`~specutils.Spectrum1D`
        The template spectrum, which will be resampled to match the wavelength
        of the observed spectrum.

    Returns
    -------
    normalized_template_spectrum : :class:`~specutils.Spectrum1D`
        The normalized spectrum template.
    chi2 : `float`
        The chi2 of the flux of the observed spectrum and the flux of the
        normalized template spectrum.
    """
    # Resample template
    if _resample(resample_method) != 0:
        fluxc_resample = _resample(resample_method)
        template_obswavelength = fluxc_resample(template_spectrum,
                                                observed_spectrum.spectral_axis)

    # Convert the uncertainty to standard deviation if needed
    stddev = _uncertainty_to_standard_deviation(observed_spectrum.uncertainty)

    # Normalize spectra
    normalization = _normalize_for_template_matching(observed_spectrum,
                                                     template_obswavelength,
                                                     stddev)

    # Numerator
    num_right = normalization * template_obswavelength.flux
    num = observed_spectrum.flux - num_right

    # Denominator
    denom = stddev * observed_spectrum.flux.unit

    # Get chi square
    result = (num/denom)**2
    chi2 = np.nansum(result.value) #Changed to np.nansum

    # Create normalized template spectrum, which will be returned with
    # corresponding chi2
    normalized_template_spectrum = Spectrum1D(
        spectral_axis=template_spectrum.spectral_axis,
        flux=template_spectrum.flux*normalization)

    return normalized_template_spectrum, chi2

def template_redshift(observed_spectrum, template_spectrum, redshift):
    """
    Find the best-fit redshift for template_spectrum to match observed_spectrum using chi2.

    Parameters
    ----------
    observed_spectrum : :class:`~specutils.Spectrum1D`
        The observed spectrum.
    template_spectrum : :class:`~specutils.Spectrum1D`
        The template spectrum, which will have it's redshift calculated.
    redshift : `float`, `int`, `list`, `tuple`, 'numpy.array`
        A scalar or iterable with the redshift values to test.

    Returns
    -------
    final_redshift : `float`
        The best-fit redshift for template_spectrum to match the observed_spectrum.
    redshifted_spectrum: :class:`~specutils.Spectrum1D`
        A new Spectrum1D object which incorporates the template_spectrum with a spectral_axis
        that has been redshifted using the final_redshift.
    chi2_list : `list`
        A list with the chi2 values corresponding to each input redshift value.
    """
    chi2_min = None
    final_redshift = None
    chi2_list = []

    redshift = np.array(redshift).reshape((np.array(redshift).size,))

    # Loop which goes through available redshift values and finds the smallest chi2
    for rs in redshift:

        # Create new redshifted spectrum and run it through the chi2 method
        redshifted_spectrum = Spectrum1D(spectral_axis=template_spectrum.spectral_axis*(1+rs),
                        flux=template_spectrum.flux, uncertainty=template_spectrum.uncertainty,
                                         meta=template_spectrum.meta)
        normalized_spectral_template, chi2 = _chi_square_for_templates(observed_spectrum, redshifted_spectrum, 
                                                                       'flux_conserving')

        chi2_list.append(chi2)

        # Set new chi2_min if suitable replacement is found - added line here b/c not returning final redshifted spectrum
        if not np.isnan(chi2) and (chi2_min is None or chi2 < chi2_min):
            chi2_min = chi2
            final_redshift = rs
            final_spectrum = redshifted_spectrum #Added

    return final_redshift, chi2_min, final_spectrum, chi2_list
    
#####################################
# Blue flexure correction functions #
#####################################
def smooth_general(lris_resolution, data_resolution, wave, flux, noise):
    """Returns flux and noise smoothed to LRIS resolution.
    
    Parameters
    ----------
    lris_resolution : tuple
        Wavelength-dependent LRIS resolutions
    data_resolution : tuple
        Resolution of the data
    wave : tuple
        Data wavelengths
    flux : tuple
        Data fluxes
    noise : tuple
        Data noise
        
    Returns
    -------
    smoothed_flux : tuple
        Data flux smoothed to LRIS resolution
    smoothed_noise : tuple
        Data noise smoothed to LRIS resolution
    """
    
    c = 299792.458 #speed of light in km/s
    sigma_aa_desired = lris_resolution/c*wave
    sigma_aa_original = data_resolution
    delta_sigma_aa_vector = np.sqrt(sigma_aa_desired**2 - sigma_aa_original**2)
    smoothed_flux = utils.smoothing.smoothspec(wave, flux, outwave=wave, smoothtype='lsf', resolution=delta_sigma_aa_vector, 
                                               fftsmooth=True)
    smoothed_noise = utils.smoothing.smoothspec(wave, noise, outwave=wave, smoothtype='lsf', resolution=delta_sigma_aa_vector, 
                                                fftsmooth=True)
    return smoothed_flux, smoothed_noise

def prep_data(data_path, spec_type):
    """Return the full blue data and the data limited to the relevant wavelength ranges.
    
    Parameters
    ----------
    data_path : str
        Path to the data file (coadd or spec1d)
    spec_type : str
        Indicates if looking at coadded spectrum or single 1D spectrum
        
    Returns
    -------
    data_wave_full : tuple
        Full blue wavelength array
    data_cut_wave : tuple
        Blue wavelength array limited to relevant range
    data_flux_full: tuple
        Full blue flux array
    data_cut_flux : tuple
        Blue flux array limited to relevant range
    data_noise_full : tuple
        Full blue noise array
    data_cut_noise : tuple
        Blue noise array limited to relevant range
    """
    
    #Import co-added spectrum with no flexure correction
    data_file = fits.open(data_path)
    if spec_type == 'coadd':
        data_wave_full = data_file[1].data['wave']
        data_flux_full = data_file[1].data['flux']
        data_noise_full = np.sqrt(data_file[1].data['ivar'])
    else:
        data_wave_full = data_file[1].data['OPT_WAVE']
        data_flux_full = data_file[1].data['OPT_COUNTS']
        data_noise_full = np.sqrt(data_file[1].data['OPT_COUNTS_IVAR'])
    data_file.close()

    #Limit to useful range
    data_cut_wave = data_wave_full[(data_wave_full >= 4000) & (data_wave_full <= 6800)]
    data_cut_flux = data_flux_full[(data_wave_full >= 4000) & (data_wave_full <= 6800)]
    data_cut_noise = data_noise_full[(data_wave_full >= 4000) & (data_wave_full <= 6800)]

    return data_wave_full, data_cut_wave, data_flux_full, data_cut_flux, data_noise_full, data_cut_noise

def prep_template(template_path):
    """Return the smoothed template data.
    
    Parameters
    ----------
    template_path : str
        Path to template spectrum file
        
    Returns
    -------
    template_wave : tuple
        Template wavelength array
    smoothed_template_flux : tuple
        Smoothed template flux array
    smoothed_template_noise : tuple
        Smoothed template noise array
    """
    
    #Import template
    template_file = pd.read_csv(template_path, skiprows=6,  sep='\t|\s+', 
                                names=['wavelength', 'flux', 'noise', 'weight', 'resolution'], index_col=None)
    template_flux = template_file['flux'].values[(template_file['wavelength'].values >= 4000) & \
                                                 (template_file['wavelength'].values <= 6800)]
    template_wave = template_file['wavelength'].values[(template_file['wavelength'].values >= 4000) & \
                                                       (template_file['wavelength'].values <= 6800)]
    template_noise = np.zeros(len(template_wave))
    template_resolution = template_file['resolution'].values[(template_file['wavelength'].values >= 4000) & \
                                                             (template_file['wavelength'].values <= 6800)]

    #Calculate desired resolution for data template (Keck resolution)
    resb_template, resr_template = lris_res(template_wave, np.nan)

    #Smooth the template 
    smoothed_template_flux, smoothed_template_noise = smooth_general(resb_template, template_resolution, template_wave, 
                                                                     template_flux, template_noise)
    return template_wave, smoothed_template_flux, smoothed_template_noise

def test_pre_full(data_path, template_path, z_lit, z_bound, spec_type):
    """Returns the full spectrum best-fit redshift and min chi2. 
    Tests the redshift array on the entire spectrum to ensure bounds are appropriate.
    
    Parameters
    ----------
    data_path : str
        Path to data spectrum file
    template_path : str
        Path to template spectrum file
    z_lit : float
        Literature redshift for target object
    z_bound : float
        Amount to add and subtract from z_lit for redshifts to test
    spec_type : str
        Indicates if looking at coadded spectrum or single 1D spectrum
        
    Returns
    -------
    tm_result[0] : float
        Best-fit redshift
    tm_result[1] : float
        Minimum chi-squared
    """
    
    #Import data
    data_wave_full, data_cut_wave, data_flux_full, data_cut_flux, data_noise_full, data_cut_noise = prep_data(data_path, 
                                                                                                              spec_type)

    #Import smoothed template
    template_wave, smoothed_template_flux, smoothed_template_noise = prep_template(template_path)

    #Continuum normalize over whole wavelength range
    norm_data_wave, norm_data_flux, norm_data_noise = continuum_normalize(np.min(data_cut_wave), np.max(data_cut_wave), 
                                                                          data_cut_flux, data_cut_wave, data_cut_noise)
    norm_template_wave, norm_template_flux, norm_template_noise = continuum_normalize(np.min(template_wave), 
                                                                                      np.max(template_wave),
                                                                                      smoothed_template_flux, template_wave, 
                                                                                      smoothed_template_noise)

    #Put spectra into Spectrum1D objects
    data_spec = Spectrum1D(spectral_axis=norm_data_wave*u.Angstrom, flux=norm_data_flux*(u.erg/u.s/u.cm**2/u.Angstrom),
                           uncertainty=StdDevUncertainty(norm_data_noise))
    template_spec = Spectrum1D(spectral_axis=norm_template_wave*u.Angstrom, flux=norm_template_flux*(u.Lsun/u.micron))

    #Plot before
    plt.figure(figsize=(12,4))
    plt.plot(data_spec.spectral_axis, data_spec.flux, label='observed')
    plt.plot(template_spec.spectral_axis, template_spec.flux, label='template')
    plt.legend()
    plt.show()

    #Fit redshifts
    redshifts = np.linspace(z_lit-z_bound, z_lit+z_bound, 1000)
    tm_result = template_redshift(observed_spectrum=data_spec, template_spectrum=template_spec, redshift=redshifts)

    #Plot after
    plt.figure(figsize=(12,4))
    plt.plot(data_spec.spectral_axis, data_spec.flux, label='observed')
    plt.plot(tm_result[2].spectral_axis, tm_result[2].flux, label='redshifted template')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(template_spec.spectral_axis, template_spec.flux, label='template')
    plt.plot(tm_result[2].spectral_axis, tm_result[2].flux, label='redshifted template')
    plt.legend()
    plt.show()
    return tm_result[0], tm_result[1]

def data_chunks(data_wave, data_flux, data_noise, targ_delta):
    """Returns normalized data spectral chunks.
    
    Parameters
    ----------
    data_wave : tuple
        Data wavelength array
    data_flux : tuple
        Data flux array
    data_noise : tuple
        Data noise array
    targ_delta : float
        Size of chunk in Angstroms
        
    Returns
    -------
    norm_wave_chunks : tuple
        Wavelength chunks of size targ_delta
    norm_flux_chunks : tuple
        Normalized flux chunks of size targ_delta
    norm_noise_chunks : tuple
        Normalized noise chunks of size targ_delta
    """
    
    #Chunk data
    #Turn arrays into lists
    wave_arr = list(data_wave)
    flux_arr = list(data_flux)
    noise_arr = list(data_noise)

    wave_chunks = []
    flux_chunks = []
    noise_chunks = []

    #Initialize counters
    i = wave_arr[0]
    k = 0
    for elem in wave_arr:
        elem_delta = elem - i
        if elem_delta >= targ_delta: #If difference is targ_delta
            end_ind = wave_arr.index(elem) + 1 #Find ending index
            wave_chunks.append(np.array(wave_arr[k:end_ind])) #Append appropriate values
            flux_chunks.append(np.array(flux_arr[k:end_ind]))
            noise_chunks.append(np.array(noise_arr[k:end_ind]))

            #Re-initialize counters
            i = wave_arr[end_ind]
            k = end_ind
    else: #Get the end of the arrays that won't be 220A wide
        wave_chunks.append(np.array(wave_arr[k:]))
        flux_chunks.append(np.array(flux_arr[k:]))
        noise_chunks.append(np.array(noise_arr[k:]))

    #Normalize chunks
    normed_data = []
    norm_wave_chunks = []
    norm_flux_chunks = []
    norm_noise_chunks = []
    for i in range(len(wave_chunks)):
        normed_data.append(continuum_normalize(np.min(wave_chunks[i]), np.max(wave_chunks[i]), flux_chunks[i], wave_chunks[i], 
                                               noise_chunks[i]))
        norm_wave_chunks.append(normed_data[i][0])
        norm_flux_chunks.append(normed_data[i][1])
        norm_noise_chunks.append(normed_data[i][2])

    return norm_wave_chunks, norm_flux_chunks, norm_noise_chunks

def template_chunks(data_wave, data_flux, data_noise, template_path, z_lit, targ_delta, overhang, position):
    """Returns normalized template spectral chunks and central wavelengths.
    
    Parameters
    ----------
    data_wave : tuple
        Data wavelength array
    data_flux : tuple
        Data flux array
    data_noise : tuple
        Data noise array
    template_path : str
        Path to template spectrum file
    z_lit : float
        Literature redshift for target object
    targ_delta : float
        Size of chunk in Angstroms
    overhang : float
        Size of overhang that template chunk should have compared to target
    position : str
        'before' or 'after' to indicate if this is pre- or post-flexure correction
        
    Returns
    -------
    norm_temp_wave_chunks : tuple
        Wavelength chunks of size targ_delta
    norm_temp_flux_chunks : tuple
        Normalized flux chunks of size targ_delta
    norm_temp_noise_chunks : tuple
        Normalized noise chunks of size targ_delta
    temp_central_wavelengths : tuple
        Central wavelength in each template chunk
    central_wavelengths : tuple
        Central wavelength in each data chunk
    """
    
    #Import smoothed template
    template_wave, smoothed_template_flux, smoothed_template_noise = prep_template(template_path)

    #Get data chunks
    data_wave_chunks, data_flux_chunks, data_noise_chunks = data_chunks(data_wave, data_flux, data_noise, targ_delta)

    #Chunk template
    #Find the central wavelength in each bin
    central_wavelengths = np.zeros(len(data_wave_chunks))
    for i in range(len(data_wave_chunks)):
        central_wavelengths[i] = np.median(data_wave_chunks[i])

    if position == 'before':
        dered_central_wavelengths = central_wavelengths/(1+z_lit)
    else:
        dered_central_wavelengths = central_wavelengths

    #Get the bounds of the template chunks (with overhang)
    template_wave_bounds = []
    for i in range(len(data_wave_chunks)):
        template_wave_bounds.append([dered_central_wavelengths[i] - (np.max(data_wave_chunks[i]) - \
                                                               np.min(data_wave_chunks[i]))/2  - overhang, 
                                     dered_central_wavelengths[i] + (np.max(data_wave_chunks[i]) - \
                                                               np.min(data_wave_chunks[i]))/2 + overhang])

    #Chunk the template based on these bounds
    template_wave_chunks = []
    template_flux_chunks = []
    template_noise_chunks = []
    for i in range(len(template_wave_bounds)):
        template_wave_chunks.append(template_wave[(template_wave >= template_wave_bounds[i][0]) & \
                                                  (template_wave <= template_wave_bounds[i][-1])])
        template_flux_chunks.append(smoothed_template_flux[(template_wave >= template_wave_bounds[i][0]) & \
                                                  (template_wave <= template_wave_bounds[i][-1])])
        template_noise_chunks.append(np.zeros(len(template_wave_chunks[i])))

    #Normalize the template chunks
    normed_template_data = []
    norm_temp_wave_chunks = []
    norm_temp_flux_chunks = []
    norm_temp_noise_chunks = []
    for i in range(len(template_wave_chunks)):
        normed_template_data.append(continuum_normalize(np.min(template_wave_chunks[i]), np.max(template_wave_chunks[i]), 
                                                        template_flux_chunks[i], template_wave_chunks[i], 
                                                        template_noise_chunks[i]))
        norm_temp_wave_chunks.append(normed_template_data[i][0])
        norm_temp_flux_chunks.append(normed_template_data[i][1])
        norm_temp_noise_chunks.append(normed_template_data[i][2])

    #Get centre of each template chunk and redshift it
    temp_central_wavelengths = np.zeros(len(norm_temp_wave_chunks))
    for i in range(len(norm_temp_wave_chunks)):
        temp_central_wavelengths[i] = np.median(norm_temp_wave_chunks[i])*(1+z_lit)

    return norm_temp_wave_chunks, norm_temp_flux_chunks, norm_temp_noise_chunks, temp_central_wavelengths, central_wavelengths

def QA_chunks(data_wave_chunks, data_flux_chunks, temp_wave_chunks, temp_flux_chunks):
    """Plots the data and template chunks against each other to make sure they look okay.
    
    Parameters
    ----------
    data_wave_chunks : tuple
        Data wavelength chunks of size targ_delta
    data_flux_chunks : tuple
        Normalized data flux chunks of size targ_delta
    temp_wave_chunks : tuple
        Template wavelength chunks of size targ_delta +- overhang
    temp_flux_chunks : tuple
        Normalized template flux chunks of size targ_delta +- overhang 
        
    Returns
    -------
    None
    """
    
    #Plot
    for i in range(len(data_wave_chunks)):
        plt.figure()
        plt.plot(data_wave_chunks[i], data_flux_chunks[i], label='observed')
        plt.plot(temp_wave_chunks[i], temp_flux_chunks[i], label='template')
        plt.legend()

def chunk_redshift(data_wave, data_flux, data_noise, template_path, z_lit, targ_delta, overhang, z_test, z_bound, position):
    """Returns the bestfit redshift of each chunk.
    
    Parameters
    ----------
    data_wave : tuple
        Data wavelength array
    data_flux : tuple
        Data flux array
    data_noise : tuple
        Data noise array
    template_path : str
        Path to template spectrum file
    z_lit : float
        Literature redshift of target object
    targ_delta : float
        Wavelength chunk size in Angstroms
    overhang : float
        Amount of wavelength overhang template chunks should have in Angstroms
    z_test : float
        Starting redshift for chunks (measured by eye in 1 chunk)
    z_bound : float
        Amount to add and subtract from z_lit for redshifts to test
    position : str
        'before' or 'after' to indicate if pre- or post-flexure correction
        
    Results
    -------
    bestfit_redshift : tuple
        Best fitting redshift for each chunk
    best_chi2 : tuple
        Minimum chi squared for each chunk
    redshifted_spectra : tuple
        Redshifted chunks
    chi2 : tuple
        All chi2
    """
    
    #Get data chunks
    data_wave_chunks, data_flux_chunks, data_noise_chunks = data_chunks(data_wave, data_flux, data_noise, targ_delta)

    #Get template_chunks 
    temp_wave_chunks, temp_flux_chunks, temp_noise_chunks, temp_central_wavelengths, central_waves = template_chunks(data_wave, 
                                                                                                                     data_flux,
                                                                                                      data_noise, template_path,
                                                                                                      z_lit, targ_delta,
                                                                                                      overhang, position)

    #Find redshifts of each chunk
    observed_chunks = []
    temp_chunks = []
    for i in range(len(data_wave_chunks)):
        observed_chunks.append(Spectrum1D(spectral_axis=data_wave_chunks[i]*u.Angstrom, 
                                          flux=data_flux_chunks[i]*(u.erg/u.s/u.cm**2/u.Angstrom), 
                                          uncertainty=InverseVariance(data_noise_chunks[i])))
        temp_chunks.append(Spectrum1D(spectral_axis=temp_wave_chunks[i]*u.Angstrom, 
                                 flux=temp_flux_chunks[i]*(u.Lsun/u.micron),
                                 uncertainty=StdDevUncertainty(temp_noise_chunks[i])))

    redshifts_chunks = np.linspace(z_test-z_bound, z_test+z_bound, 1000)
    fitted_redshift_results = []
    bestfit_redshift = np.zeros(len(data_wave_chunks))
    best_chi2 = np.zeros(len(data_wave_chunks))
    redshifted_spectra = []
    chi2 = []
    for i in range(len(data_wave_chunks)):
        fitted_redshift_results.append(template_redshift(observed_spectrum=observed_chunks[i], 
                                                            template_spectrum=temp_chunks[i],
                                                            redshift=redshifts_chunks))
        bestfit_redshift[i] = fitted_redshift_results[i][0]
        best_chi2[i] = fitted_redshift_results[i][1]
        redshifted_spectra.append(fitted_redshift_results[i][2])
        chi2.append(fitted_redshift_results[i][3])
        
    return bestfit_redshift, best_chi2, redshifted_spectra, chi2

def fit_errors(bestfit_redshift, central_wavelengths):
    """Returns the fits to be used to correct the flexure
    
    Parameters
    ----------
    bestfit_redshift : tuple
        Best fitting redshift in each wavelength chunk
    central_wavelengths : tuple
        Central wavelength of each chunk
        
    Returns
    -------
    fit : np.polyfit object
        Straight line fit to errors in Angstroms
    fit_z : np.polyfit object
        Straight line fit to errors in redshift
    fit_kms : np.polyfit object
        Straight line fit to errors in km/s
    error_func : np.poly1d object
        Straight line fit function to errors in Angstroms
    error_func_z : np.poly1d object
        Straight line fit function to errors in redshift
    error_func_kms : np.poly1d object
        Straight line fit function to errors in km/s
    errors_A : tuple
        Flexure in Angstroms
    """
    
    c = 299792.458 #speed of light in km/s
    #Calculate errors in Angstroms
    errors_A = bestfit_redshift*central_wavelengths 

    #Fit a straight line to the errors
    fit = np.polyfit(central_wavelengths, errors_A, 1)
    fit_z = np.polyfit(central_wavelengths, bestfit_redshift, 1)
    fit_kms = np.polyfit(central_wavelengths, errors_A/central_wavelengths*c, 1)

    error_func = np.poly1d(fit)
    error_func_z = np.poly1d(fit_z)
    error_func_kms = np.poly1d(fit_kms)
    
    return fit, fit_z, fit_kms, error_func, error_func_z, error_func_kms, errors_A

def QA_fit(bestfit_redshift, central_wavelengths):
    """Plots the errors and fits to check the fit visually.
    
    Parameters
    ----------
    bestfit_redshift : tuple
        Best fitting redshift in each wavelength chunk
    central_wavelengths : tuple
        Central wavelength of each chunk
    
    Returns
    -------
    None
    """
    
    fit, fit_z, fit_kms, error_func, error_func_z, error_func_kms, errors_A = fit_errors(bestfit_redshift, central_wavelengths)

    c = 299792.458 #speed of light in km/s
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    fig.set_tight_layout(True)
    #Plot in z
    ax[0].plot(central_wavelengths, bestfit_redshift, linestyle='None', marker='.', markersize=30, mew=2, mec='w', color='k',
               alpha=0.8)
    ax[0].plot(central_wavelengths, error_func_z(central_wavelengths), color='k')
    ax[0].tick_params(length=6, width=2, labelsize=12)
    ax[0].set_xlabel('Wavelength ($\AA$)', fontsize=15)
    ax[0].set_ylabel('Error (z)', fontsize=15)

    #Plot in Angstroms
    ax[1].plot(central_wavelengths, errors_A, linestyle='None', marker='.', markersize=30, mew=2, mec='w', color='k', alpha=0.8)
    ax[1].plot(central_wavelengths, error_func(central_wavelengths), color='k')
    ax[1].tick_params(length=6, width=2, labelsize=12)
    ax[1].set_xlabel('Wavelength ($\AA$)', fontsize=15)
    ax[1].set_ylabel('Error ($\Delta\AA$)', fontsize=15)

    #Plot in km/s
    ax[2].plot(central_wavelengths, errors_A/central_wavelengths*c, linestyle='None', marker='.', markersize=30, mew=2, mec='w',
               color='k', alpha=0.8)
    ax[2].plot(central_wavelengths, error_func_kms(central_wavelengths), color='k')
    ax[2].tick_params(length=6, width=2, labelsize=12)
    ax[2].set_xlabel('Wavelength ($\AA$)', fontsize=15)
    ax[2].set_ylabel('Error ($\Delta$ km/s)', fontsize=15)

def flexure_function(fit, z_lit, wave):
    """Returns flexure-corrected wavelengths.
    
    Parameters
    ----------
    fit : np.polyfit object
        Straight line fit to errors in Angstroms
    z_lit : float
        Literature redshift of target object
    wave : tuple
        Data wavelength array
    
    Returns
    -------
    corr_factor : tuple
        Amount by which to correct the wavelength array
    wave_corr : tuple
        Flexure-corrected wavelength array
    """
    
    #Solve coefficients
    c = fit[0]
    d = fit[1]

    a = d
    b = (c - z_lit)/(1+z_lit)

    corr_factor = (b*wave + a)/(1 + b)
    wave_corr = wave - corr_factor

    #Check if monotonic
    print('Montonic: ', np.all(np.diff(wave_corr) > 0))
    
    return corr_factor, wave_corr

def flexure_QA(wave, wave_corr):
    """Plots the corrected wavelength against the uncorrected to check quality of correction.
    
    Parameters
    ----------
    wave : tuple
        Uncorrected wavelength array
    wave_corr : tuple
        Flexure-corrected wavelength array
        
    Returns
    -------
    None
    """
    
    plt.figure(figsize=(8,7))
    plt.plot(wave, label='original', lw=3)
    plt.plot(wave_corr, label='corrected', linestyle='dashed')
    plt.legend()
    plt.ylabel('Wavelength ($\AA$)')
    plt.xlabel('Pixel')

def dered_corr(data_path, template_path, wave_corr, z_lit, z_bound, spec_type):
    """Function to check remaining flexure after flexure correction has been applied.
    
    Parameters
    ----------
    data_path : str
        Path to data spectrum file
    template_path : str
        Path to template spectrum file
    wave_corr : tuple
        Flexure-corrected wavelength array
    z_lit : float
        Literature redshift for target object
    z_bound : float
        Amount to add and subtract from z_lit for redshifts to test
    spec_type : str
        Indicates if looking at coadded spectrum or single 1D spectrum
        
    Returns
    -------
    tm_result_corr : float
        Best-fitting redshift post flexure-correction
    dered_wave : tuple
        De-redshifted wavelength array
    """
    
    #Get data
    data_wave_full, data_cut_wave, data_flux_full, data_cut_flux, data_noise_full, data_cut_noise = prep_data(data_path,
                                                                                                             spec_type)

    #Get template
    template_wave, smoothed_template_flux, smoothed_template_noise = prep_template(template_path)

    #Find redshift of new, corrected spectrum and de-redshift it to match the template
    #Continuum-norm over whole blue range
    norm_wave_corr, norm_corr_flux, norm_corr_noise = continuum_normalize(np.min(wave_corr), np.max(wave_corr), data_cut_flux, 
                                                                          wave_corr, data_cut_noise)
    norm_template_wave, norm_template_flux, norm_template_noise = continuum_normalize(np.min(template_wave), 
                                                                                      np.max(template_wave), 
                                                                                      smoothed_template_flux, template_wave, 
                                                                                      smoothed_template_noise)

    #Plot before
    plt.figure(figsize=(12,4))
    plt.plot(norm_wave_corr, norm_corr_flux, label='observed')
    plt.plot(norm_template_wave, norm_template_flux, label='template')
    plt.legend()

    #Find new redshift of whole spectrum
    corr_spec = Spectrum1D(spectral_axis=norm_wave_corr*u.Angstrom, flux=norm_corr_flux*(u.erg/u.s/u.cm**2/u.Angstrom),
                           uncertainty=StdDevUncertainty(norm_corr_noise))
    template_spec = Spectrum1D(spectral_axis=norm_template_wave*u.Angstrom, flux=norm_template_flux*(u.Lsun/u.micron))

    pre_redshifts = np.linspace(z_lit-z_bound, z_lit+z_bound, 1000)
    tm_result_corr = template_redshift(observed_spectrum=corr_spec, template_spectrum=template_spec, redshift=pre_redshifts)

    #Plot after
    plt.figure(figsize=(12,4))
    plt.plot(corr_spec.spectral_axis, corr_spec.flux, label='observed')
    plt.plot(tm_result_corr[2].spectral_axis, tm_result_corr[2].flux, label='redshifted template')
    plt.legend()

    plt.figure(figsize=(12,4))
    plt.plot(template_spec.spectral_axis, template_spec.flux, label='template')
    plt.plot(tm_result_corr[2].spectral_axis, tm_result_corr[2].flux, label='redshifted template')
    plt.legend()

    #De-redshift data
    dered_wave = norm_wave_corr/(1+z_lit)

    plt.figure(figsize=(12,4))
    plt.plot(dered_wave, norm_corr_flux, label='de-redshifted data')
    plt.plot(norm_template_wave, norm_template_flux, label='template')
    plt.legend()
    
    return tm_result_corr, dered_wave
    
####################################
# Red flexure correction functions #
####################################

def read_1dspec(filename):
    """Return 1D wavelengths and extracted sky spectrum.
    
    Parameters
    ----------
    filename : str
        Path to 1D spectrum
    
    Returns
    -------
    wave : tuple
        Wavelength array
    sky_flux : tuple
        Extracted sky spectrum
    """
    
    file = fits.open(filename)
    data = file[1].data
    file.close()
    
    wave = data['OPT_WAVE']
    sky_flux = data['OPT_COUNTS_SKY']
    
    return wave, sky_flux

def read_skymod(filename):
    """Return skymodel.
    
    Parameters
    ----------
    filename : str
        Skymodel file
    
    Returns 
    -------
    wave : tuple
        Wavelength array
    sky_flux : tuple
        Skymodel spectrum
    """
    
    file = fits.open(filename)
    sky_flux = file[0].data
    wave = file[1].data
    file.close()
    
    return wave, sky_flux
    
def find_nearest(array, value):
    """Return the value in an array that is closest to the given value.
    
    Parameters
    ----------
    array : tuple
        Array to examine
    value : float
        Value to examine
        
    Returns
    -------
    array[idx] : float
        Value in array that is closest to value
    """
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return array[idx]
        
def gaussian(x, a, x0, sigma, d):
    """Returns a Gaussian fit.
    
    Parameters 
    ----------
    x : tuple
        x-values
    a : float
        Stretch value
    x0 : float
        Horizontal shift value
    sigma : float
        Standard deviation
    d : float
        Vertical shift value
        
    Returns
    -------
    a*np.exp(-(x-x0)**2/(2*sigma**2)) + d : tuple
        Gaussian fit
    """
    
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + d

def data_resolution(sky_wave, sky_flux):
    """Return estimated resolution of the data.
    
    Parameters
    ----------
    sky_wave : tuple
        Wavelength array
    sky_flux : tuple
        Flux array
    
    Returns
    -------
    R_red : float
        Estimated resolution
    """
    
    #Find FWHM of 4 lines across the spectrum and take the mean to find the resolution 
    line1 = [5575, 5585]
    line2 = [6300, 6305]
    line3 = [8346, 8348]
    line4 = [9377, 9380]

    line1_wave, line1_flux, line1_noise = continuum_normalize(line1[0], line1[-1], sky_flux, sky_wave, np.zeros_like(sky_flux))
    n_line1 = len(line1_wave)
    mean_line1 = np.mean(line1_wave)
    std_line1 = np.std(line1_wave)
    popt_line1, pcov_line1 = curve_fit(gaussian, line1_wave, line1_flux, p0=[np.max(line1_flux), mean_line1, std_line1, 0])
    FWHM_line1 = 2*np.sqrt(2*np.log(2))*np.abs(popt_line1[2])

    line2_wave, line2_flux, line2_noise = continuum_normalize(line2[0], line2[-1], sky_flux, sky_wave, np.zeros_like(sky_flux))
    n_line2 = len(line2_wave)
    mean_line2 = np.mean(line2_wave)
    std_line2 = np.std(line2_wave)
    popt_line2, pcov_line2 = curve_fit(gaussian, line2_wave, line2_flux, p0=[np.max(line2_flux), mean_line2, std_line2, 0])
    FWHM_line2 = 2*np.sqrt(2*np.log(2))*np.abs(popt_line2[2])

    line3_wave, line3_flux, line3_noise = continuum_normalize(line3[0], line3[-1], sky_flux, sky_wave, np.zeros_like(sky_flux))
    n_line3 = len(line3_wave)
    mean_line3 = np.mean(line3_wave)
    std_line3 = np.std(line3_wave)
    popt_line3, pcov_line3 = curve_fit(gaussian, line3_wave, line3_flux, p0=[np.max(line3_flux), mean_line3, std_line3, 0])
    FWHM_line3 = 2*np.sqrt(2*np.log(2))*np.abs(popt_line3[2])

    line4_wave, line4_flux, line4_noise = continuum_normalize(line4[0], line4[-1], sky_flux, sky_wave, np.zeros_like(sky_flux))
    n_line4 = len(line4_wave)
    mean_line4 = np.mean(line4_wave)
    std_line4 = np.std(line4_wave)
    popt_line4, pcov_line4 = curve_fit(gaussian, line4_wave, line4_flux, p0=[np.max(line4_flux), mean_line4, std_line4, 0])
    FWHM_line4 = 2*np.sqrt(2*np.log(2))*np.abs(popt_line4[2])
    
    R_red = np.mean((FWHM_line1, FWHM_line2, FWHM_line3, FWHM_line4))
    
    return R_red

def smooth_red(sky_wave, sky_flux):
    """Return sky spectrum smoothed to data resolution.
    
    Parameters
    ----------
    sky_wave : tuple
        Wavelength array
    sky_flux : tuple
        Flux array
    
    Returns
    -------
    smoothed_sky_flux : tuple
        Smoothed flux array
    smoothed_sky_noise : tuple
        Smoothed noise array
    """
    
    desired_blue_res, desired_red_res = lris_res(sky_wave[sky_wave <= 7000], sky_wave[sky_wave > 7000])
    desired_res = np.concatenate((desired_blue_res, desired_red_res))
    c = 299792.458 #speed of light
    
    #Smooth red 
    R_red = data_resolution(sky_wave, sky_flux)
    in_sigma_kms_red = (R_red/sky_wave)*c
    smoothed_sky_flux, smoothed_sky_noise = smooth_general(desired_res, in_sigma_kms_red/c*sky_wave, sky_wave, sky_flux, 
                                                           np.zeros_like(sky_flux))
    
    return smoothed_sky_flux, smoothed_sky_noise

def sky_centroids(sky_wave, smoothed_sky_flux, smoothed_sky_noise):
    """Returns centroids of skylines in the sky model.
    
    Parameters
    ----------
    sky_wave : tuple
        Wavelength array
    smoothed_sky_flux : tuple
        Smoothed flux array
    smoothed_sky_noise : tuple
        Smoothed noise array
        
    Returns
    -------
    sky_cents : tuple
        Skyline centroids
    """
    
    #Normalize sky model
    sky_norm_wave, sky_norm_flux, sky_norm_noise = continuum_normalize(np.min(sky_wave), np.max(sky_wave), smoothed_sky_flux, 
                                                                       sky_wave, smoothed_sky_noise)
    
    #Find centroids of skylines in model
    centroids = [7843, 7916, 7995.5, 8347, 8467, 8829.5, 8922, 9378, 9442, 9794]
    sky_bounds = [[7835, 7850], [7908, 7920], [7990, 8000], [8340, 8353], [8460, 8475], [8820, 8835], [8915, 8930], 
                      [9370, 9385], [9435, 9447], [9785, 9799]]

    sky = Spectrum1D(spectral_axis=sky_norm_wave*u.Angstrom, flux=sky_norm_flux*u.ct)
    regions_sky = []
    for i in range(len(sky_bounds)):
        regions_sky.append(SpectralRegion(sky_bounds[i][0]*u.Angstrom, sky_bounds[i][-1]*u.Angstrom))

    sky_cents = centroid(sky, regions_sky)
    
    return sky_cents

def centroid_offsets(targ_bounds, data_wave, data_flux, sky_cents):
    """Returns amount by which extracted skylines are offset from model and the nearest wavelength value to each.
    
    Parameters
    ----------
    targ_bounds : tuple
        List of tuples defining bounds of region around each skyline to examine
    data_wave : tuple
        Wavelength array
    data_flux : tuple
        Flux array
    sky_cents : tuple
        Skymodel centroids
        
    Returns
    -------
    nearest_waves : tuple
        Nearest wavelength value to centroid
    offsets : tuple
        Offset between data and skymodel
    """
    
    regions = SpectralRegion(targ_bounds[0][0]*u.Angstrom,targ_bounds[0][-1]*u.Angstrom)
    for i in range(1, len(targ_bounds)):
        regions += SpectralRegion(targ_bounds[i][0]*u.Angstrom, targ_bounds[i][-1]*u.Angstrom)

    #Normalize data
    targ_norm_wave, targ_norm_flux, targ_norm_noise = continuum_normalize(np.min(data_wave), np.max(data_wave), data_flux, 
                                                                          data_wave, np.zeros(len(data_flux)))
    
    
    #Find offsets
    target = Spectrum1D(spectral_axis=targ_norm_wave*u.Angstrom, flux=targ_norm_flux*u.ct)
    sub_spec = extract_region(target, regions)
    offsets = np.zeros(len(sky_cents))
    nearest_waves = np.zeros(len(sky_cents))
    for i, sub in enumerate(sub_spec):
        an_disp = sub.flux.max()
        an_ampl = sub.flux.min()
        an_mean = sub.spectral_axis[sub.flux.argmax()]
        nearest_waves[i] = an_mean.value
        an_stdv = np.sqrt(np.sum((sub.spectral_axis - an_mean)**2) / (len(sub.spectral_axis) - 1))

        plt.figure()
        plt.scatter(an_mean.value, an_disp.value, marker='o', color='#e41a1c', s=100, label='data')
        plt.scatter(sky_cents[i], an_disp.value, marker='o', color='k', s=100, label='archive')
        plt.vlines([an_mean.value - an_stdv.value, an_mean.value + an_stdv.value],
                    sub.flux.min().value, sub.flux.max().value,
                    color='#377eb8', ls='--', lw=2)
        g_init = ( models.Const1D(an_disp) +
                  models.Gaussian1D(amplitude=(an_ampl - an_disp),
                                mean=an_mean, stddev=an_stdv) )
        g_fit = fit_lines(sub, g_init)
        line_fit = g_fit(sub.spectral_axis)
        plt.plot(sub.spectral_axis, sub.flux, color='#e41a1c', lw=2)
        plt.plot(sub.spectral_axis, line_fit, color='#377eb8', lw=2)

        plt.axvline(an_mean.value, color='#e41a1c', ls='--', lw=2)
        plt.legend()
        offsets[i] = an_mean.value - sky_cents[i].value
        
    return nearest_waves, offsets

def offset_fit(nearest_waves, offsets):
    """Return straight line fit to offsets, to be applied to correct flexure.
    
    Parameters
    ----------
    nearest_waves : tuple
        Nearest wavelength value to centroid
    offsets : tuple
        Offset between data and skymodel
    
    Returns
    -------
    fit : np.polyfit object
        Fit to offsets in Angstroms
    fit_kms : np.polyfit object
        Fit to offsets in km/s
    fit_func : np.poly1d object
        Fit function in Angstroms
    fit_func_kms : np.poly1d object
        Fit function in km/s
    """
    
    c = 299792.458 #speed of light
    fit = np.polyfit(nearest_waves, offsets, 1)
    fit_kms = np.polyfit(nearest_waves, offsets/nearest_waves*c, 1)

    fit_func = np.poly1d(fit)
    fit_func_kms = np.poly1d(fit_kms)
    
    return fit, fit_kms, fit_func, fit_func_kms

def QA_offset_fit(nearest_waves, offsets):
    """Plot the offsets and fits to check fit.
    
    Parameters
    ----------
    nearest_waves : tuple
        Nearest wavelength value to centroid
    offsets : tuple
        Offset between data and skymodel
        
    Returns
    -------
    None
    """
    
    fit, fit_kms, fit_func, fit_func_kms = offset_fit(nearest_waves, offsets)
    c = 299792.458 #speed of light
    
    #Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    fig.set_tight_layout(True)
    
    #In Angstroms
    ax[0].plot(nearest_waves, offsets, linestyle='None', marker='.', markersize=30, mew=2, mec='w', alpha=0.8, color='k')
    ax[0].plot(nearest_waves, fit_func(nearest_waves), color='k', alpha=0.8)
    ax[0].tick_params(length=6, width=2, labelsize=12)
    ax[0].set_xlabel('Wavelength ($\AA$)', fontsize=15)
    ax[0].set_ylabel('Error ($\AA$)', fontsize=15)
    
    #In km/s
    ax[1].plot(nearest_waves, offsets/nearest_waves*c, linestyle='None', marker='.', markersize=30, mew=2, mec='w', 
             alpha=0.8, color='k')
    ax[1].plot(nearest_waves, fit_func_kms(nearest_waves), color='k', alpha=0.8)
    ax[1].tick_params(length=6, width=2, labelsize=12)
    ax[1].set_xlabel('Wavelength ($\AA$)', fontsize=20)
    ax[1].set_ylabel('Error (km/s)', fontsize=20)
    
def fit_interp(data_wave, data_flux, data_noise, nearest_waves, fit_func):
    """Return the flexure-corrected wavelengths.
    
    Parameters
    ----------
    data_wave : tuple
        Wavelength array
    data_flux : tuple
        Flux array
    data_noise : tuple
        Noise array
    nearest_waves : tuple
        Nearest wavelengths to centroids
    fit_func : np.poly1d object
        Fit to offsets in Angstroms
        
    Returns
    -------
    wave_corr : tuple
        Flexure-corrected wavelength array 
    """
    
    #Interpolate fits over wavelength range
    error_func = interp1d(nearest_waves, fit_func(nearest_waves), fill_value='extrapolate')
    corr_factor = error_func(data_wave)
    
    #Add correction factor to wavelength array
    wave_corr = data_wave - corr_factor
    
    #Make sure new wavelength arrays are monotonically increasing
    plt.figure(figsize=(8,7))
    plt.plot(data_wave, label='original', lw=3)
    plt.plot(wave_corr, label='corrected', linestyle='dashed')
    plt.xlabel('Pixel')
    plt.legend()
    plt.ylabel('Wavelength ($\AA$)')

    print('Monotonic: ', np.all(np.diff(wave_corr) > 0))
    
    return wave_corr
