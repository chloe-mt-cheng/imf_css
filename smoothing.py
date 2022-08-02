import telluric_correction as tell
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftfreq, rfftfreq
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

def spikes(wave, flux, hi, lo, mid, mid2):
    '''Return indices where large telluric spikes/artifacts are.

    Parameters
    ----------
    wave : tuple
      Wavelength array
    flux : tuple
      Flux array
    hi : float
      Threshold for how high spikes need to be in order to mask them
    lo : float
      Threshold for how low spikes need to be in order ot mask them
    mid : float
      Threshold for spikes from telluric correction (where the blue and red side join)
    mid2 : float
      Sometimes need another threshold for telluric spikes <= 7500A

    Returns
    -------
    spikes_hi : tuple
      Positions of spikes above hi threshold
    spikes_lo : tuple
      Positions of spikes below lo threshold
    spikes_mid : tuple
      Positions of spikes above mid threshold
    spikes_mid2 : tuple
      Positions of spikes above mid2 threshold
    '''

    spikes_hi = np.argwhere(flux > hi)
    spikes_lo = np.argwhere(flux < lo)

    #Get mid-spectra spikes
    inds = []
    for i in range(len(flux)):
        if wave[i] >= 7500 and wave[i] <= 8000:
            inds.append(i)
            
    inds2 = []
    for i in range(len(flux)):
        if wave[i] >= 6800 and wave[i] <= 7500:
            inds2.append(i)

    spikes_mid = []
    for i in range(len(flux[inds])):
        if flux[inds][i] >= mid:
            spikes_mid.append(inds[i])
    
    spikes_mid2 = []
    for i in range(len(flux[inds2])):
        if flux[inds2][i] >= mid2:
            spikes_mid2.append(inds2[i])

    return spikes_hi, spikes_lo, spikes_mid, spikes_mid2

#######################
#Modified Prospector smoothspec functions (added correct noise smoothing)
def mask_wave(wavelength, width=1, wlo=0, whi=np.inf, outwave=None,
              nsigma_pad=20.0, linear=False, **extras):
    """Restrict wavelength range (for speed) but include some padding based on
    the desired resolution.
    """
    # Base wavelength limits
    if outwave is not None:
        wlim = np.array([outwave.min(), outwave.max()])
    else:
        wlim = np.squeeze(np.array([wlo, whi]))
    # Pad by nsigma * sigma_wave
    if linear:
        wlim += nsigma_pad * width * np.array([-1, 1])
    else:
        wlim *= (1 + nsigma_pad / width * np.array([-1, 1]))
    mask = (wavelength > wlim[0]) & (wavelength < wlim[1])
    return mask

def smooth_fft(dx, spec, sigma):
    """Basic math for FFT convolution with a gaussian kernel.
    :param dx:
        The wavelength or velocity spacing, same units as sigma
    :param sigma:
        The width of the gaussian kernel, same units as dx
    :param spec:
        The spectrum flux vector
    """
    # The Fourier coordinate
    ss = rfftfreq(len(spec), d=dx)
    # Make the fourier space taper; just the analytical fft of a gaussian
    taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
    ss[0] = 0.01  # hack
    # Fourier transform the spectrum
    spec_ff = np.fft.rfft(spec)
    # Multiply in fourier space
    ff_tapered = spec_ff * taper
    # Fourier transform back
    spec_conv = np.fft.irfft(ff_tapered)
    return spec_conv

def smooth_fft_noise(dx, spec, sigma):
    """Basic math for FFT convolution with a gaussian kernel.
    :param dx:
        The wavelength or velocity spacing, same units as sigma
    :param sigma:
        The width of the gaussian kernel, same units as dx
    :param spec:
        The spectrum flux vector
    """
    # The Fourier coordinate
    ss = rfftfreq(len(spec), d=dx)
    # Make the fourier space taper; just the analytical fft of a gaussian
    taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
    ss[0] = 0.01  # hack
    # Fourier transform the spectrum
    spec_ff = np.fft.rfft(spec)
    # Multiply in fourier space
    ff_tapered = spec_ff * taper**2
    # Fourier transform back
    spec_conv = np.fft.irfft(ff_tapered)
    return spec_conv

def smooth_lsf_fft(wave, spec, unc, outwave, b, sigma=None, lsf=None, pix_per_sigma=2,
                   eps=0.25, preserve_all_input_frequencies=False, corr=False, **kwargs):
    """Smooth a spectrum by a wavelength dependent line-spread function, using
    FFTs.
    :param wave:
        Wavelength vector of the input spectrum.
    :param spectrum:
        Flux vector of the input spectrum.
    :param unc:
        Uncertainty vector of the input spectrum.
    :param outwave:
        Desired output wavelength vector.
    :param sigma: (optional)
        Dispersion (in same units as ``wave``) as a function `wave`.  ndarray
        of same length as ``wave``.  If not given, sigma will be computed from
        the function provided by the ``lsf`` keyword.
    :param lsf: (optional)
        Function used to calculate the dispersion as a function of wavelength.
        Must be able to take as an argument the ``wave`` vector and any extra
        keyword arguments and return the dispersion (in the same units as the
        input wavelength vector) at every value of ``wave``.  If not provided
        then ``sigma`` must be specified.
    :param pix_per_sigma: (optional, default: 2)
        Number of pixels per sigma of the smoothed spectrum to use in
        intermediate interpolation and FFT steps. Increasing this number will
        increase the accuracy of the output (to a point), and the run-time, by
        preserving all high-frequency information in the input spectrum.
    :param preserve_all_input_frequencies: (default: False)
        This is a switch to use a very dense sampling of the input spectrum
        that preserves all input frequencies.  It can significantly increase
        the call time for often modest gains...
    :param eps: (optional)
        Deprecated.
    :param corr: (default: False)
        Indicates whether the data is correlated or not.  Default is 
        uncorrelated.  If the data is uncorrelated, the variance of the 
        convolved data is found by convolving the variance of the data with the
        square of the kernel used to convolve the data.  If the data is correlated, 
        the variance of the convolved data is found by convolving the variance of 
        the data with the square of the Gaussian kernel, but scaled with a factor
        depending on the widths of the Gaussian kernel and width of the Gaussian 
        coefficient in the covariance.  Uncertainty propagation method from 
        R. Klein 2021.
    :param **kwargs:
        All additional keywords are passed to the function supplied to the
        ``lsf`` keyword, if present.
    :returns flux:
        The input spectrum smoothed by the wavelength dependent line-spread
        function.  Same length as ``outwave``.
    """
    # This is sigma vs lambda
    if sigma is None:
        sigma = lsf(wave, **kwargs)

    # Now we need the CDF of 1/sigma, which provides the relationship between x and lambda
    # does dw go in numerator or denominator?
    # I think numerator but should be tested
    dw = np.gradient(wave)
    cdf = np.cumsum(dw / sigma)
    cdf /= cdf.max()

    # Now we create an evenly sampled grid in the x coordinate on the interval [0,1]
    # and convert that to lambda using the cdf.
    # This should result in some power of two x points, for FFT efficiency

    # Furthermore, the number of points should be high enough that the
    # resolution is critically sampled.  And we want to know what the
    # resolution is in this new coordinate.
    # There are two possible ways to do this

    # 1) Choose a point ~halfway in the spectrum
    # half = len(wave) / 2
    # Now get the x coordinates of a point eps*sigma redder and bluer
    # wave_eps = eps * np.array([-1, 1]) * sigma[halpha]
    # x_h_eps = np.interp(wave[half] + wave_eps, wave, cdf)
    # Take the differences to get dx and dsigma and ratio to get x per sigma
    # x_per_sigma = np.diff(x_h_eps) / (2.0 * eps) #x_h_epsilon - x_h

    # 2) Get for all points (slower?):
    sigma_per_pixel = (dw / sigma)
    x_per_pixel = np.gradient(cdf)
    x_per_sigma = np.nanmedian(x_per_pixel / sigma_per_pixel)
    N = pix_per_sigma / x_per_sigma

    # Alternatively, just use the smallest dx of the input, divided by two for safety
    # Assumes the input spectrum is critically sampled.
    # And does not actually give x_per_sigma, so that has to be determined anyway
    if preserve_all_input_frequencies:
        # preserve more information in the input spectrum, even when way higher
        # frequency than the resolution of the output.  Leads to slightly more
        # accurate output, but with a substantial time hit
        N = max(N, 1.0 / np.nanmin(x_per_pixel))

    # Now find the smallest power of two that divides the interval (0, 1) into
    # segments that are smaller than dx
    nx = int(2**np.ceil(np.log2(N)))

    # now evenly sample in the x coordinate
    x = np.linspace(0, 1, nx)
    dx = 1.0 / nx

    # And now we get the spectrum at the lambda coordinates of the even grid in x
    lam = np.interp(x, cdf, wave)
    newspec = np.interp(lam, wave, spec)
    newunc = np.interp(lam, wave, unc)

    # And now we convolve.
    # If we did not know sigma in terms of x we could estimate it here
    # from the resulting sigma(lamda(x)) / dlambda(x):
    # dlam = np.gradient(lam)
    # sigma_x = np.median(lsf(lam, **kwargs) / dlam)
    # But the following just uses the fact that we know x_per_sigma (duh).
    spec_conv = smooth_fft(dx, newspec, x_per_sigma)
    
    #If the data is correlated
    if corr:
        theta = x_per_sigma #width of the d-dimensional Gaussian kernel (should be the same as sigma fed to smooth_fft?)
        b = b #width of the Gaussian coefficient in the covariance??? Set to be some number close to 1 for now
        d = len(x) #d-dimensional Gaussian (number of points)
        unc_conv = ((2*np.sqrt(pi)*theta*b)/np.sqrt(theta**2 + b**2))**d*smooth_fft_noise(dx, newunc, x_per_sigma)
    #If the data is uncorrelated
    else:
        unc_conv = smooth_fft_noise(dx, newunc, x_per_sigma)

    # and interpolate back to the output wavelength grid.
    smooth_spec = np.interp(outwave, lam, spec_conv)
    smooth_unc = np.interp(outwave, lam, unc_conv)
    
    return smooth_spec, smooth_unc

def smoothspec(wave, spec, unc, b, resolution=None, outwave=None,
               smoothtype="vel", fftsmooth=True,
               min_wave_smooth=0, max_wave_smooth=np.inf, corr=False, **kwargs):
    """
    Parameters
    ----------
    wave : ndarray of shape ``(N_pix,)``
        The wavelength vector of the input spectrum.  Assumed Angstroms.
    spec : ndarray of shape ``(N_pix,)``
        The flux vector of the input spectrum.
    unc : ndarray of shape ``(N_pix,)``
        The uncertainty vector of the input spectrum.  Assumed variance.
    resolution : float
        The smoothing parameter.  Units depend on ``smoothtype``.
    outwave : ``None`` or ndarray of shape ``(N_pix_out,)``
        The output wavelength vector.  If ``None`` then the input wavelength
        vector will be assumed, though if ``min_wave_smooth`` or
        ``max_wave_smooth`` are also specified, then the output spectrum may
        have different length than ``spec`` or ``wave``, or the convolution may
        be strange outside of ``min_wave_smooth`` and ``max_wave_smooth``.
        Basically, always set ``outwave`` to be safe.
    smoothtype : string, optional, default: "vel"
        The type of smoothing to perform.  One of:
        + ``"vel"`` - velocity smoothing, ``resolution`` units are in km/s
          (dispersion not FWHM)
        + ``"R"`` - resolution smoothing, ``resolution`` is in units of
          :math:`\lambda/ \sigma_\lambda` (where :math:`\sigma_\lambda` is
          dispersion, not FWHM)
        + ``"lambda"`` - wavelength smoothing.  ``resolution`` is in units of
          Angstroms
        + ``"lsf"`` - line-spread function.  Use an aribitrary line spread
          function, which can be given as a vector the same length as ``wave``
          that gives the dispersion (in AA) at each wavelength.  Alternatively,
          if ``resolution`` is ``None`` then a line-spread function must be
          present as an additional ``lsf`` keyword.  In this case all additional
          keywords as well as the ``wave`` vector will be passed to this ``lsf``
          function.
    fftsmooth : bool, optional, default: True
        Switch to use FFTs to do the smoothing, usually resulting in massive
        speedups of all algorithms.  However, edge effects may be present.
    min_wave_smooth : float, optional default: 0
        The minimum wavelength of the input vector to consider when smoothing
        the spectrum.  If ``None`` then it is determined from the output
        wavelength vector and padded by some multiple of the desired resolution.
    max_wave_smooth : float, optional, default: inf
        The maximum wavelength of the input vector to consider when smoothing
        the spectrum.  If None then it is determined from the output wavelength
        vector and padded by some multiple of the desired resolution.
    inres : float, optional
        If given, this parameter specifies the resolution of the input.  This
        resolution is subtracted in quadrature from the target output resolution
        before the kernel is formed.
        In certain cases this can be used to properly switch from resolution
        that is constant in velocity to one that is constant in wavelength,
        taking into account the wavelength dependence of the input resolution
        when defined in terms of lambda.  This is possible iff:
        * ``fftsmooth`` is False
        * ``smoothtype`` is ``"lambda"``
        * The optional ``in_vel`` parameter is supplied and True.
        The units of ``inres`` should be the same as the units of
        ``resolution``, except in the case of switching from velocity to
        wavelength resolution, in which case the units of ``inres`` should be
        in units of lambda/sigma_lambda.
    in_vel : float (optional)
        If supplied and True, the ``inres`` parameter is assumed to be in units
        of lambda/sigma_lambda. This parameter is ignored **unless** the
        ``smoothtype`` is ``"lambda"`` and ``fftsmooth`` is False.
    Returns
    -------
    flux : ndarray of shape ``(N_pix_out,)``
        The smoothed spectrum on the `outwave` grid, ndarray.
    """
    if smoothtype == 'vel':
        linear = False
        units = 'km/s'
        sigma = resolution
        fwhm = sigma * sigma_to_fwhm
        Rsigma = ckms / sigma
        R = ckms / fwhm
        width = Rsigma
        assert np.size(sigma) == 1, "`resolution` must be scalar for `smoothtype`='vel'"

    elif smoothtype == 'R':
        linear = False
        units = 'km/s'
        Rsigma = resolution
        sigma = ckms / Rsigma
        fwhm = sigma * sigma_to_fwhm
        R = ckms / fwhm
        width = Rsigma
        assert np.size(sigma) == 1, "`resolution` must be scalar for `smoothtype`='R'"
        # convert inres from Rsigma to sigma (km/s)
        try:
            kwargs['inres'] = ckms / kwargs['inres']
        except(KeyError):
            pass

    elif smoothtype == 'lambda':
        linear = True
        units = 'AA'
        sigma = resolution
        fwhm = sigma * sigma_to_fwhm
        Rsigma = None
        R = None
        width = sigma
        assert np.size(sigma) == 1, "`resolution` must be scalar for `smoothtype`='lambda'"

    elif smoothtype == 'lsf':
        linear = True
        width = 100
        sigma = resolution

    else:
        raise ValueError("smoothtype {} is not valid".format(smoothtype))

    # Mask the input spectrum depending on outwave or the wave_smooth kwargs
    mask = mask_wave(wave, width=width, outwave=outwave, linear=linear,
                     wlo=min_wave_smooth, whi=max_wave_smooth, **kwargs)
    w = wave[mask]
    s = spec[mask]
    u = unc[mask]
    if outwave is None:
        outwave = wave

    # Choose the smoothing method
    if smoothtype == 'lsf':
        if fftsmooth:
            smooth_method = smooth_lsf_fft
            if sigma is not None:
                # mask the resolution vector
                sigma = resolution[mask]
        else:
            smooth_method = smooth_lsf
            if sigma is not None:
                # convert to resolution on the output wavelength grid
                sigma = np.interp(outwave, wave, resolution)
    elif linear:
        if fftsmooth:
            smooth_method = smooth_wave_fft
        else:
            smooth_method = smooth_wave
    else:
        if fftsmooth:
            smooth_method = smooth_vel_fft
        else:
            smooth_method = smooth_vel

    # Actually do the smoothing and return
    smooth_spec, smooth_unc = smooth_method(w, s, u, outwave, b, sigma, corr, **kwargs)
    return smooth_spec, smooth_unc
#####################

def smooth_chunk(wave, flux, noise, res, spike_vals, literature_sigma, smooth_value, b, corr, smooth_type):
    """Return smoothed spectra, where regions of large telluric spikes have been interpolated over.
    Smooth in chunks defined outside of the function (so you can smooth blue and red separately)

    Parameters
    ----------
    wave : tuple
      Wavelength array
    flux : tuple
      Tellurically-corrected flux
    noise : tuple
      Uncertainty array (variance)
    res : tuple
      Wavelength-dependent instrumental resolution from Keck
    spike_vals : tuple
      Spike threshold values to mask
    literature_sigma : float
      Velocity dispersion from the literature
    smooth_value : float
      Value to smooth by/to
    b : float
      For correlated noise, not relevant anymore (only use uncorrelated version)
    corr : bool
      True if noise is correlated, False if uncorrelated
    smooth_type : str
      'by' or 'to 

    Returns
    -------
    wave : tuple
      Wavelength array
    smoothed_flux : tuple
      Final, smoothed flux, to be fit with alf
    np.sqrt(smoothed_noise) : tuple
      Smoothed noise
    bad_pixels : tuple
      List of pixels being interpolated over
    flux_cut : tuple
      Flux array where we've removed regions < 4000A and around 6300A.  For testing. 
    res : tuple
      Wavelength-dependent LRIS resolution
    """

    flux_cut = np.copy(flux)
    
    #Get pixel positions of noisy telluric spikes
    hi, lo, mid, mid2 = spike_vals
    spikes_hi, spikes_lo, spikes_mid, spikes_mid2 = spikes(wave, flux, hi, lo, mid, mid2)
    bad_pixels = []
    for i in range(len(spikes_hi)):
        for j in range(len(spikes_hi[i])):
            bad_pixels.append(spikes_hi[i][j])
    for i in range(len(spikes_lo)):
        for j in range(len(spikes_lo[i])):
            bad_pixels.append(spikes_lo[i][j])
    for i in range(len(spikes_mid)):
        bad_pixels.append(spikes_mid[i])
    for i in range(len(spikes_mid2)):
        bad_pixels.append(spikes_mid2[i])

    bad_pixels = np.sort(list(set(bad_pixels))) #Remove duplicates
    
    #Interpolate over noisy spikes
    if len(bad_pixels) != 0:
        flux[bad_pixels] = np.nan
        noise[bad_pixels] = np.nan

    kernel = Gaussian1DKernel(stddev = 1)
    intp_flux = interpolate_replace_nans(flux, kernel, convolve = convolve_fft)
    intp_noise = interpolate_replace_nans(noise, kernel, convolve = convolve_fft)

    #Smooth to smooth_value
    c = 299792.458 #speed of light
    if smooth_type == 'to':
        sigma_aa_desired = smooth_value/c*wave #Convert to Angstroms
        sigma_aa_original = np.sqrt(res**2 + literature_sigma**2)/c*wave #Effective resolution converted to Angstroms
        delta_sigma_aa_vector = np.sqrt(sigma_aa_desired**2 - sigma_aa_original**2)
        plt.plot(wave, sigma_aa_desired**2 - sigma_aa_original**2)
    elif smooth_type == 'by':
        sigma_aa_desired = smooth_value/c*wave
        delta_sigma_aa_vector = sigma_aa_desired
        plt.plot(wave, sigma_aa_desired)
    
    smoothed_flux, smoothed_noise = smoothspec(wave, intp_flux, intp_noise, outwave=wave, b=1.05, smoothtype='lsf', 
                                               resolution=delta_sigma_aa_vector, fftsmooth=True, corr=corr)
    
    return wave, smoothed_flux, np.sqrt(smoothed_noise), bad_pixels, flux_cut, res
    