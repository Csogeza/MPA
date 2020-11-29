import numpy as np
import yaml
import astropy.units as u
import astropy
import math
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from scipy.integrate import quad, trapz


def get_rhoph_powerlaw(n1, v_ph, mu_e, t):
    v_ph = (v_ph * u.kilometer /(1 * u.s)).to("cm / s")
    mu_e = mu_e * u.gram
    sig_e = astropy.constants.sigma_T.to("cm2")
    t = (t * u.day).to(u.s)

    return 2/3 * mu_e * (n1 - 1) / (sig_e * t * v_ph)


def get_rhoph(n1, n2, v_ph, v_th, mu_e, t):
    v_ph = (v_ph * u.kilometer /(1 * u.s)).to("cm / s")
    v_th = (v_th * u.kilometer /(1 * u.s)).to("cm / s")
    mu_e = mu_e * u.gram
    sig_e = astropy.constants.sigma_T.to("cm2")
    t = (t * u.day).to(u.s)
	
	
    if v_ph < v_th:
        return 2/3 * mu_e/(sig_e * t * v_ph) / (1/(n1-1) +
		(v_ph / v_th)**(n1-1) * (1/(n2-1) - 1/(n1-1)))
    else:
        return 2/3 * mu_e * (n2 - 1) / (sig_e * t * v_ph)


def get_vinner_powerlaw(n1, v_ph, mu_e, tau, rho_ph, t):
    
    v_ph = (v_ph * u.kilometer / (1 * u.s)).to("cm / s")
    mu_e = mu_e * u.gram
    sig_e = astropy.constants.sigma_T.to("cm2")
    t = (t * u.day).to(u.s)
    
    first_t = tau * mu_e * (n1 - 1) / (rho_ph * sig_e * t * v_ph**n1)
        
    try:
        return (np.power(first_t, 1 / (1 - n1))).to("km / s")

    except astropy.units.core.UnitConversionError:
        return (np.power(first_t.value, 1 / (1 - n1)) * u.cm / (1 * u.s)).to("km / s")		


def get_vinner(n1, n2, v_ph, v_th, mu_e, tau, rho_ph, t):
    
    v_ph = (v_ph * u.kilometer / (1 * u.s)).to("cm / s")
    v_th = (v_th * u.kilometer / (1 * u.s)).to("cm / s")
    mu_e = mu_e * u.gram
    sig_e = astropy.constants.sigma_T.to("cm2")
    t = (t * u.day).to(u.s)
    
    if v_ph < v_th:
        first_t = tau * mu_e * (n1 - 1) / (rho_ph * sig_e * t * v_ph) / v_ph**(n1 - 1)
        second_t = 1 / v_th**(n1 - 1) * ( (n1 - 1) / (n2 - 1) - 1)
        
    else:
        first_t = tau * mu_e * (n1 - 1) / (rho_ph * sig_e * t) * v_th**(n2 - n1) / v_ph**(n2)
        second_t = 1 / v_th**(n1 - 1) * ( (n1 - 1) / (n2 - 1) - 1)
        
    try:
        return (np.power(first_t - second_t, 1 / (1 - n1))).to("km / s")

    except astropy.units.core.UnitConversionError:
        return (np.power(first_t.value - second_t.value, 1 / (1 - n1)) * u.cm / (1 * u.s)).to("km / s")
    
    # Astropy seems not to be able to raise the cm^n / s^n to the 1/n th power if n is a float

	
def calc_mu(X, Y):

    return 1.6735575e-24 * (X + 1/4 * Y)**-1

def calc_mu_mod(X, Y, ZHeEff):

    return 1.6735575e-24 * (X + ZHeEff * 1/4 * Y)**-1
     
    # In case we want to set up a radius dependent mu_e profile; warning: not complete procedure
    # it only works properly with a given set of parameters (although still better approximation
    # than the constant mu_e case)
    
def mu_eShift(vth):

    coeffs = np.array([ 1.40153223e-08, -3.48015261e-04,  2.65275189e+00, -6.36722042e+03])
    shift = np.polyval(coeffs,vth)
    
    return shift

    # Match an average mu_e profile to our case which is described by the v_th in an empirical
    # sense; just shift this profile in accordance with the v_th
def mu_eProfile(shift):

    av_prof = np.load('mu_e_profile.npy')
    
    st_vls = np.linspace(3000, min(av_prof[:,0]), 100)
    st_mus = np.full(100, calc_mu_mod(0.548822580264,0.425148381550787,1))
    st_prof_vl = np.append(st_vls, av_prof[:,0])
    st_prof_mu = np.append(st_mus, av_prof[:,1])
    
    s = InterpolatedUnivariateSpline(av_prof[:,0], av_prof[:,1], k=3)
    extr_vl = np.linspace(14000, 15000, 10)
    
    mu_prof_vl = np.append(st_prof_vl, extr_vl)
    mu_prof_mu = np.append(st_prof_mu, s(extr_vl))
    
    return np.vstack((mu_prof_vl+shift, mu_prof_mu)).T	

    
    # The program that is calculates rho_ph for a given mu_e profile
def get_rhoph_mod(n1, n2, v_ph, v_th, mu_e_prof, t):
    v_ph = (v_ph * u.kilometer /(1 * u.s)).to("cm / s").value
    v_th = (v_th * u.kilometer /(1 * u.s)).to("cm / s").value
    mu_e_prof[:,0] = (mu_e_prof[:,0] * u.kilometer /(1 * u.s)).to("cm / s").value
    sig_e = astropy.constants.sigma_T.to("cm2").value
    t = (t * u.day).to(u.s).value
    
    if v_ph < v_th:
        func1 = interp1d(mu_e_prof[:,0], 1/np.multiply(np.power(mu_e_prof[:,0],n2), mu_e_prof[:,1]))
        func2 = interp1d(mu_e_prof[:,0], 1/np.multiply(np.power(mu_e_prof[:,0],n1), mu_e_prof[:,1]))

        int1 = -1 * quad(func1, v_th, max(mu_e_prof[:,0]))[0]
        int2 = -1 * quad(func2, v_ph, v_th)[0]
        
        return 2/3 / (v_ph**n1 * sig_e * t * (-1 * v_th**(n2-n1) * int1 - int2)) * u.g / u.cm**3
    
    else:
        func1 = interp1d(mu_e_prof[:,0], 1/np.multiply(np.power(mu_e_prof[:,0],n2), mu_e_prof[:,1]))

        int1 = quad(func1, v_ph, max(mu_e_prof[:,0]))[0]

        return 2/3 / (v_ph**n2 * sig_e * t * int1) * u.g / u.cm**3
     
     
    
def get_vinner_mod(n1, n2, v_ph, v_th, mu_e_prof, tau, rho_ph, t):
    v_ph = (v_ph * u.kilometer / (1 * u.s)).to("cm / s").value
    v_th = (v_th * u.kilometer / (1 * u.s)).to("cm / s").value
    mu_e_prof[:,0] = (mu_e_prof[:,0] * u.kilometer /(1 * u.s)).to("cm / s").value
    #mu_e = mu_e * u.gram
    sig_e = astropy.constants.sigma_T.to("cm2").value
    t = (t * u.day).to(u.s).value
    rho_ph = rho_ph.value
    
    
    if v_ph < v_th:
        
        def cost_func(v_inner):            
            func1 = interp1d(mu_e_prof[:,0], 1/np.multiply(np.power(mu_e_prof[:,0],n2), mu_e_prof[:,1]))
            func2 = interp1d(mu_e_prof[:,0], 1/np.multiply(np.power(mu_e_prof[:,0],n1), mu_e_prof[:,1]))
            
            first_term = tau / (rho_ph * v_ph**n1 * sig_e * t)
            second_term = 1 / (v_th**(n1-n2)) * quad(func1, v_th, max(mu_e_prof[:,0]))[0]
            third_term = quad(func2, v_inner, v_th)[0]
            
            return 1e21*(first_term - second_term - third_term)
        
    else:
        
        def cost_func(v_inner):            
            func1 = interp1d(mu_e_prof[:,0], 1/np.multiply(np.power(mu_e_prof[:,0],n2), mu_e_prof[:,1]))
            func2 = interp1d(mu_e_prof[:,0], 1/np.multiply(np.power(mu_e_prof[:,0],n1), mu_e_prof[:,1]))
            
            first_term = tau / (rho_ph * v_ph**n2 * sig_e * t)
            second_term = quad(func1, v_th, max(mu_e_prof[:,0]))[0]
            third_term = 1 / (v_th**(n2-n1)) * quad(func2, v_inner, v_th)[0]
            
            return 1e21*(first_term - second_term - third_term)
           
            
    return (root_scalar(cost_func,method='brentq',bracket=[4000*1e5,6000*1e5]).root * u.cm / (1 * u.s)).to("km / s")	
    
    
def generate_powerlaw_grid_vph(
    n1,
    v_ph,
    t0,
    mu_e,
    tau,
    vgrid_n,
    v_grid_max
):

    rho_ph = get_rhoph_powerlaw(n1, v_ph, mu_e, t0)
    v_inner = get_vinner_powerlaw(n1, v_ph, mu_e, tau, rho_ph, t0)
    
    vgrid = np.logspace(math.log(v_inner.value,3), math.log(v_grid_max,3), vgrid_n)
    vgrid = ((vgrid - min(vgrid)) / (max(vgrid) - min(vgrid))) * (v_grid_max - v_inner.value) + v_inner.value 

    vgrid_mids = np.zeros(vgrid.shape)
    vg_diffs = np.ediff1d(vgrid)
    vgrid_mids[0] = vgrid[0] - vg_diffs[0]/2
    vgrid_mids[1:] = vgrid[1:] - vg_diffs/2

    rhogrid = np.zeros(vgrid.shape)
    #cond = vgrid_mids <= v_th

    rhogrid = rho_ph.value * np.power(v_ph / vgrid_mids, n1)
    
    return np.vstack((vgrid, rhogrid)).T

	
	
def get_breakoff_powerlaw(vr_grid, breakoff):
    
    rhogrid = vr_grid.copy()
    
    total_dens = trapz(rhogrid[:,1], rhogrid[:,0])
    rhogrid[rhogrid[:,0] > breakoff, 1] = 0
    
    new_dens = trapz(rhogrid[:,1], rhogrid[:,0])
    
    new_rho = rhogrid[:,1] / (new_dens / total_dens)
    
    return np.vstack((rhogrid[:,0],new_rho)).T
	
	
def get_breakoff_powerlaw_mod(vr_grid, breakoff):
    
    rhogrid = vr_grid.copy()
    
    total_dens = trapz(rhogrid[:,1], rhogrid[:,0])
    rhogrid[rhogrid[:,0] > breakoff, 1] = 0
    
    new_dens = trapz(rhogrid[:,1], rhogrid[:,0])
    
    new_rho = rhogrid[:,1] / (new_dens / total_dens)
    
    return np.vstack((rhogrid[rhogrid[:,1] > 0,0],new_rho[rhogrid[:,1] > 0])).T
	
	
	
def write_out_breakoff_profile(vr_prof, n1, v_ph, v_br, t0, mu_e, tau,
    templ_file="tardis_pow_cust_new_H.yml", outp_yml="tardis_broken_pow_cust_new.yml", drs = '',
    T_inner = 13750, n_threads = 1, n_packets = 2000000, n_last_packets = 2000000):
    
    rho_ph = get_rhoph_powerlaw(n1, v_ph, mu_e, t0)
    v_inner = get_vinner_powerlaw(n1, v_ph, mu_e, tau, rho_ph, t0)
	
    with open(
            drs+"power-density-n1_"
            + str(n1)
            + "-t"
            + str(t0)
            + "-vph_"
            + str(v_ph)
            + "-vbr_"
            + str(v_br)
            + "-mu_e_"
            + str(mu_e)
            + "-T_inn_"
            + str(T_inner)
            + ".dat",
            "w",
        ) as f:
            f.write("%f day\n" % t0)
            f.write("# index velocity (km/s) density (g/cm^3)\n")
            for i in range(vr_prof.shape[0]):
                f.write("%i\t%.3f\t%.7e\n" % (i, vr_prof[i,0], vr_prof[i,1]))
    
    with open(templ_file) as f:
            yob = yaml.load(f)
    yob["model"]["structure"]["filename"] = (
            drs+"power-density-n1_"
            + str(n1)
            + "-t"
            + str(t0)
            + "-vph_"
            + str(v_ph)
            + "-vbr_"
            + str(v_br)
            + "-mu_e_"
            + str(mu_e)
            + "-T_inn_"
            + str(T_inner)
            + ".dat"
        )
    yob["model"]["structure"]["v_inner_boundary"] = (
            str(round(v_inner.value, 5)) + " km/s"
        )
        
    yob["supernova"]["time_explosion"] = str(round(t0, 10)) + " day"
    yob["montecarlo"]["nthreads"]  = n_threads
    yob["montecarlo"]["last_no_of_packets"]  = n_last_packets
    yob["montecarlo"]["no_of_packets"]  = n_packets
    yob["plasma"]["initial_t_inner"] = str(T_inner) + " K"
    
    with open(outp_yml, "w") as f:
            yaml.dump(yob, f)
	
	
def generate_broken_powerlaw_grid_vph(
    n1,
    n2,
    v_th,
    v_ph,
    t0,
    mu_e,
    tau,
    vgrid_n,
    v_grid_max,
    templ_file="tardis_pow_cust_new_H.yml",
    outp_yml="tardis_broken_pow_cust_new.yml",
	drs = '',
    T_inner = 13750,
    n_threads = 1,
    n_packets = 2000000,
    n_last_packets = 2000000
):

    #shft = mu_eShift(v_th)
    #mu_e_prof = mu_eProfile(shft)

    rho_ph = get_rhoph(n1, n2, v_ph, v_th, mu_e, t0)
    #rho_ph = get_rhoph_mod(n1, n2, v_ph, v_th, mu_e_prof.copy(), t0)
    v_inner = get_vinner(n1, n2, v_ph, v_th, mu_e, tau, rho_ph, t0)
    #v_inner = get_vinner_mod(n1, n2, v_ph, v_th, mu_e_prof.copy(), tau, rho_ph, t0)

    # The density is in g/cm3, while the velocity is in km/s. From here on the units
    # will be omitted, as they do not matter in the calculations

    #vgrid = np.linspace(v_inner.value, v_grid_max, vgrid_n)   # Depending on what velocity sampling do we prefer
    
    #vgrid = np.logspace(np.log10(v_inner.value), np.log10(v_grid_max), vgrid_n)
    
    vgrid = np.logspace(math.log(v_inner.value,3), math.log(v_grid_max,3), vgrid_n)
    vgrid = ((vgrid - min(vgrid)) / (max(vgrid) - min(vgrid))) * (v_grid_max - v_inner.value) + v_inner.value 
    
    #vgrid_mids = vgrid - (vgrid[1] - vgrid[0]) / 2
    
    vgrid_mids = np.zeros(vgrid.shape)
    vg_diffs = np.ediff1d(vgrid)
    vgrid_mids[0] = vgrid[0] - vg_diffs[0]/2
    vgrid_mids[1:] = vgrid[1:] - vg_diffs/2

    rhogrid = np.zeros(vgrid.shape)
    cond = vgrid_mids <= v_th

    if v_ph < v_th:

        rhogrid[cond] = rho_ph.value * np.power(v_ph / vgrid_mids[cond], n1)
        rhogrid[~cond] = (
            rho_ph.value
            * np.power(v_ph / v_th, n1)
            * np.power(v_th / vgrid_mids[~cond], n2)
        )

    else:

        rhogrid[cond] = (
            rho_ph.value
            * np.power(v_ph / v_th, n2)
            * np.power(v_th / vgrid_mids[cond], n1)
        )
        rhogrid[~cond] = rho_ph.value * np.power(v_ph / vgrid_mids[~cond], n2)

    # Printing to a file with a custom name based on the parameters of the model

    with open(
        drs+"power-density-n1_"
        + str(n1)
        + "-n2_"
        + str(n2)
        + "-t"
        + str(t0)
        + "-vph_"
        + str(v_ph)
        + "-vth_"
        + str(v_th)
        + "-mu_e_"
        + str(mu_e)
        + "-T_inn_"
        + str(T_inner)
        + ".dat",
        "w",
    ) as f:
        f.write("%f day\n" % t0)
        f.write("# index velocity (km/s) density (g/cm^3)\n")
        for i in range(vgrid.shape[0]):
            f.write("%i\t%.3f\t%.7e\n" % (i, vgrid[i], rhogrid[i]))

    with open(templ_file) as f:
        yob = yaml.load(f)
    yob["model"]["structure"]["filename"] = (
        drs+"power-density-n1_"
        + str(n1)
        + "-n2_"
        + str(n2)
        + "-t"
        + str(t0)
        + "-vph_"
        + str(v_ph)
        + "-vth_"
        + str(v_th)
        + "-mu_e_"
        + str(mu_e)
        + "-T_inn_"
        + str(T_inner)
        + ".dat"
    )
    yob["model"]["structure"]["v_inner_boundary"] = (
        str(round(v_inner.value, 5)) + " km/s"
    )
    # yob['model']['abundances']['H'] = str(X)
    # yob['model']['abundances']['He'] = str(Y)
    yob["supernova"]["time_explosion"] = str(round(t0, 10)) + " day"
    yob["montecarlo"]["nthreads"]  = n_threads
    yob["montecarlo"]["last_no_of_packets"]  = n_last_packets
    yob["montecarlo"]["no_of_packets"]  = n_packets
    yob["plasma"]["initial_t_inner"] = str(T_inner) + " K"

    with open(outp_yml, "w") as f:
        yaml.dump(yob, f)
