import numpy as np
from pyDOE import lhs
from BPLDens import DensityProfile as DP
import astropy.constants
import astropy.units as u

# Determine the velocity from an optical depth value
def calc_vth_from_tau(n2, tau, mu_e, v_ph, rho_ph, t):
    # However, this works only if we ensure that v_th < v_ph, otherwise we need a more
    # complex mehtod for the determination of the photospheric density
    
    v_ph = (v_ph * u.kilometer / (1 * u.s)).to("cm / s")
    mu_e = mu_e * u.gram
    sig_e = astropy.constants.sigma_T.to("cm2")
    t = (t * u.day).to(u.s)
    
    temp = sig_e * t * rho_ph / (mu_e * tau * (n2 - 1)) * v_ph**n2
    
    return numpy.power(temp, 1 / (n2 - 1)).to("km / s")


# Transform the [0,1] range to the specified one (only uniform currently, except for tau - v)
def transformScales(sample, parameter_ints, n2, mu_e, v_ph, t):
    
    for i in range (len(parameter_ints)):
        sample[:,i] = (parameter_ints[i][1] - parameter_ints[i][0]) * sample[:,i] + parameter_ints[i][0]
    
    # If the second column is now considered as the tau, we have to transform it a bit further
    # to get the threshold velocity
    
    rho = DP.get_rhoph(0, n2, v_ph, 0, mu_e, t)
    
    for j in range (sample.shape[0]):
        sample[j,1] = calc_vth_from_tau(n2, sample[j,1], mu_e, v_ph, rho, t).value


# Generate the grid with LHS, then get a smaller grid with random numbers for testing if necessary
def generateGrid(N, parameter_ints, n2, mu_e, v_ph, t, test = False):
    unif_sample = lhs(len(parameter_ints), samples = N)
    
    transformScales(unif_sample, parameter_ints, n2, mu_e, v_ph, t)
        
    if test: # Then generate some test cases with completely random dist
        test_sample = np.random.rand(int(N / 5), len(parameter_ints))
        
        transformScales(test_sample, parameter_ints, n2, mu_e, v_ph, t)
        
        return [unif_sample, test_sample]
        
    else:
        return [unif_sample]