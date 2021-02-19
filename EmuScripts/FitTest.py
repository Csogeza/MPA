from EmuScripts.emulator import *
from EmuScripts.covariance import FixedWhiteNoise, CompoundCovariance
import numpy as np
import pandas as pd

df_ref = pd.read_csv('./df_ref_for_test_em.csv')

cut_list = get_filelist_below_temp(7500)

emu_list = []
test_list = []

test_inds = [25,26,28,31,32]

for i in range (len(cut_list)):
	if i in test_inds:
		test_list.append(cut_list[i])
	else:
		emu_list.append(cut_list[i])
		
res = parse_pms_new(filelist = emu_list, n_components = 20, shift = False,
 wav_range=(3000, 11000))

temp = pd.read_hdf(test_list[0], key="wav") 
cond = (temp > 3500) & (temp < 10500)

Emu = T_v_emulator(res, temp[cond], shift = False, telluric_mask='full')
bounds = Emu.initialize_bounds_from_prior()

for i in range (len(test_list)):
	Emu.spec = pd.read_hdf(test_list[i], key="lum").values[cond]

	white_cov = FixedWhiteNoise.from_spectrum(Emu.spec)

	cov = CompoundCovariance([white_cov])

	Emu._covariance = cov

	ml = Emu.do_max_like_fit('DE')
	
	ml.to_hdf('./Fits/2d/NOC_test_'+test_list[i].split('_')[-1])
	
	x_set = pd.Series(df_ref.iloc[i].values[2:],['T_ph','n1'])
	
	test_spec = Emu.emulate(x_set)
	
	np.save('./Fits/2d/NOC_test_emulated_'+test_list[i].split('_')[-1],
            np.vstack((Emu.wav,test_spec)).T)
	