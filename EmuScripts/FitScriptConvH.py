from EmuScripts.emulator import *
from EmuScripts.covariance import FixedWhiteNoise, CompoundCovariance
import numpy as np

cut_list = get_filelist_below_temp(7500)

emu_list = []
test_list = []

test_inds = [25,26,28,31,32]

for i in range (len(cut_list)):
	if i in test_inds:
		test_list.append(cut_list[i])
	else:
		emu_list.append(cut_list[i])
		
res = parse_pms_new(filelist = emu_list, n_components = 20, shift = True, wav_range=(3000, 11000))

spec_data = np.loadtxt('ext_shifted_1999em_nov14.dat')

Emu = T_v_emulator(res, spec_data[:,0], shift=True, telluric_mask='full')

Emu.spec = spec_data[:,1]

Emu.setup_hulls()

white_cov = FixedWhiteNoise.from_spectrum(Emu.spec)

cov = CompoundCovariance([white_cov])

Emu.covariance = cov

bounds = Emu.initialize_bounds_from_prior()

n1s = np.linspace(6,10,9)
for i in range (n1s.shape[0]):
	Emu.n_preset = n1s[i]
	ml = Emu.do_max_like_fit('DE')
	ml.to_hdf('./Fits/2d/ConvH_vph_fit_n1-'+str(i+1)+'.hdf')
	