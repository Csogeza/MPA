from BPLDens import DensityProfile as BPL
from BPLDens import ModelGridGenerator as MGG

################################################################################################################
# SET THE PARAMETERS BELOW
################################################################################################################


n2 = 10      # This is fixed, the other is drawn randomly
v_ph = 7794  # Photospheric velocity
tau = 18.54  # At the inner boundary
t = 18.5355  # Time since explosion

parameter_ints = [[6, 10], [2/3, 2.5], [0, 1]]    # This is where we modify the parameter intervals

N_gen = 10                # Number of spectra to be simulated

Nds = 60                  # The number of point pairs in the density profile files

vg_max = 14250            # The end of the velocity grid

test_bool = False         # If true, a test set will be generated too

X = 0.548822580264        # Abundancies
Y = 0.425148381550787
Zeff = 0.5470185894631937 # The ionization fraction of helium, necessary correction factor obtained by num. int.


################################################################################################################


mu_e = BPL.calc_mu_mod(X, Y, Zeff)

samples = generateGrid(N_gen, parameter_ints, n2, mu_e, v_ph, t, test = test_bool)


################################################################################################################
# Now that we have the uniform sample in the multi dimensional field of the physical parameters, we can generate
# the source files for the simulations
################################################################################################################

for i in range (sample[0].shape[0]):
    BPL.generate_broken_powerlaw_grid_vph(
        samples[0][i,0],
        n2,
        samples[0][i,1],
        v_ph,
        t,
        mu_e,
        tau,
        Nds,
        vg_max,
        templ_file = 'model_template.yml',
        outp_yml='SimFiles/Grid/Emulate/model_'+str(i+1)+'.yml',
        drs='SimFiles/Grid/Emulate/',
        T_inner = samples[0][i,2],
        n_threads = 1,
        n_packets = 4000000
    )
    
    
if test_bool:
    for i in range (sample[1].shape[0]):
        BPL.generate_broken_powerlaw_grid_vph(
            samples[1][i,0],
            n2,
            samples[1][i,1],
            v_ph,
            t,
            mu_e,
            tau,
            Nds,
            vg_max,
            templ_file = 'model_template.yml',
            outp_yml='SimFiles/Grid/Test/model_'+str(i+1)+'.yml',
            drs='SimFiles/Grid/Test/',
            T_inner = samples[1][i,2],
            n_threads = 1,
            n_packets = 4000000
        )
    
    
    
   