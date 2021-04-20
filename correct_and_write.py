import numpy as np
import pandas as pd
import telluric_correction as tell
import smoothing

def correct_and_write(blue_path, red_path, hi, lo, mid, fname):
	tell_wave, tell_flux = tell.telluric_corrections(blue_path, red_path)
	smooth_wave, smooth_flux, smooth_noise = smoothing.smoothing(blue_path, red_path, tell_wave, tell_flux, hi, lo, mid)
	resolution_col = np.zeros(len(smooth_wave))
	weight_col = np.ones(len(smooth_wave))
	
	df = pd.DataFrame(np.array((smooth_wave, smooth_flux, smooth_noise, weight_col, resolution_col)).T)
	df.to_csv(fname + '.dat', index=False, sep='\t')