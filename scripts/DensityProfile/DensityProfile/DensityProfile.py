import numpy as np
import yaml
import astropy.units as u
import astropy


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
    n_packets = 2000000
):

    rho_ph = get_rhoph(n1, n2, v_ph, v_th, mu_e, t0)
    v_inner = get_vinner(n1, n2, v_ph, v_th, mu_e, tau, rho_ph, t0)

    # The density is in g/cm3, while the velocity is in km/s. From here on the units
    # will be omitted, as they do not matter in the calculations

    vgrid = np.linspace(v_inner.value, v_grid_max, vgrid_n)
    vgrid_mids = vgrid - (vgrid[1] - vgrid[0]) / 2
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
    yob["montecarlo"]["no_of_packets"]  = n_packets
    yob["plasma"]["initial_t_inner"] = str(T_inner) + " K"

    with open(outp_yml, "w") as f:
        yaml.dump(yob, f)
