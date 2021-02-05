from EmuScripts.emulator import *
from EmuScripts.covariance import FixedWhiteNoise, CompoundCovariance
import numpy as np
import pandas as pd

cut_list = get_filelist_below_temp(7500)

emu_list = []
test_list = []

test_inds = [25,26,28,31,32]

for i in range (len(cut_list)):
	if i in test_inds:
		test_list.append(cut_list[i])
	else:
		emu_list.append(cut_list[i])
		
res = parse_pms_new(filelist = emu_list, n_components = 20, shift = True, v_shift=7160,
 wav_range=(3000, 11000))

temp = pd.read_hdf(test_list[0], key="wav") 
cond = (temp > 3500) & (temp < 10500)

Emu = T_v_emulator(res, temp[cond], shift = True, v_shift=7160, telluric_mask='full')
bounds = Emu.initialize_bounds_from_prior()

for i in range (len(test_list)):
	Emu.spec = pd.read_hdf(test_list[i], key="lum").values[cond]

	white_cov = FixedWhiteNoise.from_spectrum(Emu.spec)

	cov = CompoundCovariance([white_cov])

	Emu._covariance = cov

	ml = Emu.do_max_like_fit('DE')
	
	ml.to_hdf('./Fits/FixC_test_'+test_list[i].split('_')[-1])
	
	
	