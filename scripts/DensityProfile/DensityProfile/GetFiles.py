from BPLDens import DensityProfile as BPL

n1 = 10
n2 = 14
v_ph = 7794
tau = 18.54
t = 18.5355
N = 30
vg_max = 14250
mu_e = BPL.calc_mu(1.0,0)

# Setup grid
v_th_grid =  [ 6450.,  7450.,  8450.,  9450., 10450., 11450., 12450., 14250.]

for i in range (len(v_th_grid)):
	BPL.generate_broken_powerlaw_grid_vph(n1, n2, v_th_grid[i], v_ph, t, mu_e, tau, N, vg_max, \
	templ_file = 'model_template_test.yml', outp_yml='SimFiles/model_'+str(v_th_grid[i])+'.yml', drs='SimFiles/')