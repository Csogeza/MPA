import os
import re
import yaml
import warnings
import numpy as np
import pandas as pd
#from observation import Reddening
from astropy import units, constants as const
from phot_util import calculate_grid_magnitudes
from util import (
    shift_wavelength, get_v_phot,
    get_git_revision_hash, get_files_from_paths,
    HDFSaverMixin, DiffEvoCallBack, get_exponent
)
# from backports import tempfile
# from pymultinest.solve import Solver
from stats_util import UniformPrior, CompoundPrior
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, brute, minimize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, FunctionTransformer)
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, Matern,
    ConstantKernel as C
)
from scipy.spatial import Delaunay
from collections import namedtuple, OrderedDict
from numpy.random import RandomState

my_path = os.path.abspath(os.path.dirname(__file__))

re_Z = re.compile('Z(?P<Z_value>[0-9]\.[0-9]{2,})') # noqa

telluric_bands = {
    'none': [],
    'NaD': [[5870, 5920]],
    'full': [[5870, 5920], [6820, 6890], [7530, 7670]]
}

CCM_AS_AV_ratio = {'B': 1.337, 'R': 0.751, 'V': 1.0, 'I': 0.479}

PCA_Decomp = namedtuple(
    'PCA_Decomp', ['data', 'data_norm', 'pca', 'ts',
                   'wav_cent', 'eigenvalues', 'data_orig',
                   'params_frame', 'mags']
)


BaseMaxLikeResult = namedtuple(
    'MaxLikeResult', ['wav', 'spec', 'max_like_spec', 'params',
                      'thetas', 'theta_vs', 'd_theta_vs_d_RV']
)


class MaxLikeResult(BaseMaxLikeResult, HDFSaverMixin):
    pass

BaseMaxLikeResult_tr = namedtuple(
    'MaxLikeResult_tr', ['wav', 'spec', 'max_like_spec', 'params']
)

class MaxLikeResult_tr(BaseMaxLikeResult_tr, HDFSaverMixin):
    pass

sav_order = 5  # Previously 3
print "Using savgol order {}".format(sav_order)

wav_path = os.path.join(my_path, 'util_data', 'wav.hdf')

def in_hull(p, hull):
    return hull.find_simplex(p) >= 0

def parse_params(path='./', abunds=None, parse_density=False,
                 ref_path=None, files=None, index_by_fname=False,
                 constraints=None):
    if files is None:
        files = get_files_from_paths(
            path, constraints=constraints
        )
    try:
        Ts = []
        for fname in files:
            with pd.HDFStore(fname, mode='r') as hdf:
                Ts.append(hdf.t_inner[0])
    except: # noqa
        print "Resort to extraction of Ts from file name"
        try:
            Ts = [float((fname.split('-T')[1]).split(
                  '.hdf')[0]) for fname in files]
        except: # noqa
            Ts = [float((fname.split('-T')[1]).split(
                  '-v')[0]) for fname in files]
    if abunds:
        Xdata = np.zeros((len(files), len(abunds)))
        X_frame = pd.DataFrame(Xdata, columns=abunds)
    else:
        X_frame = None
    if parse_density:
        ndata = np.zeros(len(files))
    else:
        ndata = None
    i = 0
    if re.search(re_Z, files[0]):
        Z = []
        has_Z = True
    else:
        has_Z = False

    ts = []
    T_phs = []
    vs = []
    for hdf_name, T in zip(files, Ts):
        with pd.HDFStore(hdf_name, mode='r') as hdf:
            if abunds:
                X1 = get_abundances(
                        hdf_name, abunds, ref_path
                )
                X_frame.ix[i] = X1
            if parse_density:
                ndata[i] = get_exponent(
                        hdf_name, ref_path
                )
            i += 1
            v, T_ph = get_v_phot(hdf)
            T_phs.append(T_ph)
            vs.append(v)
            t = hdf.time_explosion[0] * units.s
            if has_Z:
                Z.append(float(
                    re.search(
                        re_Z, hdf_name).group('Z_value'))
                )
            ts.append(t.to(units.day).value)

    ts = np.array(ts)
    if has_Z:
        Z = np.array(Z)
    else:
        Z = None

    params_dict = {'v': vs, 'T': Ts, 'T_ph': T_phs}
    if has_Z:
        params_dict.update({'Z': Z})
    if parse_density:
        params_dict.update({'n': ndata})
    if not len(np.unique(ts)) == 1:
        params_dict.update({'ts': ts})
    params_frame = pd.DataFrame.from_dict(params_dict)
    if index_by_fname:
        index = files
        params_frame.index = index
    return params_frame


# TODO: Use parse_params
def parse_t_v_grid_shift(path='./', n_components=3, wav_range=(2500, 10000),
                         abunds=None, parse_density=False,
                         ref_path=None, files=None, renorm_wavelength=6000.,
                         constraints=None):
    #import pdb; pdb.set_trace()
    if files is None:
        files = get_files_from_paths(path, constraints=constraints)
    try:
        Ts = []
        for fname in files:
            with pd.HDFStore(fname, mode='r') as hdf:
                Ts.append(hdf.t_inner[0])
    except: # noqa
        print "Resort to extraction of Ts from file name"
        try:
            Ts = [float((fname.split('-T')[1]).split('.hdf')[0])
                  for fname in files]
        except: # noqa
            Ts = [float((fname.split('-T')[1]).split('-v')[0])
                  for fname in files]

    wav_path = os.path.join(my_path, 'util_data', 'wav.hdf')
    with pd.HDFStore(wav_path, mode='r') as hdwav:
        wav = hdwav['wav']

    wav_cent = (wav.values[:-1] + wav.values[1:])/2.

    wav_mask = np.logical_and(wav_cent > wav_range[0],
                              wav_cent < wav_range[1])

    wav_cent_mask = wav_cent[wav_mask]

    index = pd.MultiIndex(levels=[[], []],
                          labels=[[], []], names=[u'v', u'T'])

    data = pd.DataFrame(columns=np.arange(wav_mask.sum()), index=index)
    data_orig = pd.DataFrame(columns=np.arange(wav_mask.sum()), index=index)

    T_phs = []
    ts = []
    if abunds:
        Xdata = np.zeros((len(files), len(abunds)))
        X_frame = pd.DataFrame(Xdata, columns=abunds)
    else:
        X_frame = None
    if parse_density:
        ndata = np.zeros(len(files))
    else:
        ndata = None
    i = 0
    if re.search(re_Z, files[0]):
        Z = []
        has_Z = True
    else:
        has_Z = False
    for hdf_name, T in zip(files, Ts):
        with pd.HDFStore(hdf_name, mode='r') as hdf:
            if abunds:
                X1 = get_abundances(hdf_name, abunds, ref_path)
                X_frame.ix[i] = X1
            if parse_density:
                ndata[i] = get_exponent(hdf_name, ref_path)
            i += 1
            v, T_ph = get_v_phot(hdf)
            T_phs.append(T_ph)
            t = hdf.time_explosion[0] * units.s
            if has_Z:
                Z.append(float(
                    re.search(re_Z, hdf_name).group('Z_value'))
                )
            ts.append(t.to(units.day).value)
            spec = savgol_filter(hdf['lum'], 21, sav_order)

            data_orig.loc[(v, T), :] = spec[wav_mask]
            new_wav = shift_wavelength(wav_cent, v * (units.km / units.s))
            new_flux = interp1d(new_wav, spec, fill_value='extrapolate')(
                wav_cent_mask
            )
            data.loc[(v, T), :] = new_flux

    renorm_ref = interp1d(wav_cent_mask, data)(renorm_wavelength)
    norm = data.divide(renorm_ref, axis=0)

    v = norm.index.get_level_values('v')
    T = norm.index.get_level_values('T')
    ts = np.array(ts)
    if has_Z:
        Z = np.array(Z)
    else:
        Z = None

    # ('log', FunctionTransformer(np.log, inverse_func=np.exp)),
    pca = Pipeline(steps=[
        ('scaler', MinMaxScaler(feature_range=(-0.5, 0.5))),
        # ('scaler', FunctionTransformer(lambda x: x)),
        ('pca', PCA(n_components=n_components))
    ])
    pca.fit(norm.values)
    eigenvalues = pca.fit_transform(norm.values)

    # v = np.log(v)
    params_dict = {'v': v, 'T': T, 'T_ph': T_phs}
    if has_Z:
        params_dict.update({'Z': Z})
    if parse_density:
        # params_dict.update({'n': np.log(ndata)})
        params_dict.update({'n': ndata})
    if not len(np.unique(ts)) == 1:
        params_dict.update({'ts': ts})
    params_frame = pd.DataFrame.from_dict(params_dict)
    # TODO: Add abunds to params_frame
    mags = calculate_grid_magnitudes(files)
    correction = 5. * np.log10(v.values * ts * (units.day.to(units.s)) / 10. * (
            units.km.to(units.pc)))
    mags = mags.add(correction, axis=0)
    return PCA_Decomp(
            data=data, data_norm=norm, wav_cent=wav_cent_mask,
            pca=pca, eigenvalues=eigenvalues, data_orig=data_orig,
            mags=mags,
            params_frame=params_frame, ts=ts,
    )

# -----------------------------------------------------------------------------
# Modified script that takes the parameters from a physical parameter file

def get_filelist_below_temp(T_lim, sim_dir = '/afs/mpa/data/csogeza/SimResults/2dNewEpoch/'):
    fl_list = [i for i in os.listdir(sim_dir) if i.endswith('.hdf')]

    cut_list = []
    for i in fl_list:
        with pd.HDFStore(sim_dir+i, mode = 'r') as hdf:
            v, T_ph = get_v_phot(hdf)
            if T_ph < T_lim:
                cut_list.append(sim_dir + i)

    return cut_list

def get_wav_vals(wav_range):
    
    wav_path = os.path.join(my_path, 'util_data', 'wav.hdf')
    
    with pd.HDFStore(wav_path, mode='r') as hdwav:
        wav = hdwav['wav']

    wav_cent = (wav.values[:-1] + wav.values[1:])/2.

    wav_mask = np.logical_and(wav_cent > wav_range[0],
                                  wav_cent < wav_range[1])

    wav_cent_mask = wav_cent[wav_mask]
    
    return wav_cent_mask, wav_mask, wav_cent


# The set of parameters here are hardcoded to T_inner, T_ph, n1, tau_ph, v
def parse_pms_new(path='./', n_components=3, wav_range=(2500, 10000), shift=False, v_shift=None,
                         renorm_wavelength=6000., constraints=None, filelist=None): 

    if filelist is None:
        files = get_files_from_paths(path, constraints=constraints)

    else:
        files = filelist  # But these need to have the complete paths, or it won't work
    
    Ts = [pd.read_hdf(fname, key = "t_inner")[0] for fname in files]
    inds = [int(fname.split('_')[-1].split('.')[0]) for fname in files]
    
    wav_cent_mask, wav_mask, wav_cent = get_wav_vals(wav_range)
    
    #index = pd.MultiIndex(levels=[[], []],
    #                          labels=[[], []], names=[u'v', u'T'])
    index = pd.Index(files)
    data = pd.DataFrame(columns=np.arange(wav_mask.sum()))
    data_orig = pd.DataFrame(columns=np.arange(wav_mask.sum()), index=index)
    
    T_phs = []
    v_phs = []
    ts = []
    
    for hdf_name, T in zip(files, Ts):
        with pd.HDFStore(hdf_name, mode='r') as hdf:
            v, T_ph = get_v_phot(hdf)
            T_phs.append(T_ph)
            v_phs.append(v)
            t = hdf.time_explosion[0] * units.s

            ts.append(t.to(units.day).value)
            spec = savgol_filter(hdf['lum'], 21, sav_order)

            #data_orig.loc[(hdf_name), :] = spec[wav_mask]
            
            # First attempt: apply no velocity shifting at all
            new_wav = wav_cent
            if shift == True:
                if v_shift is None:
                    new_wav = shift_wavelength(wav_cent, v * (units.km / units.s))
                else:
                    new_wav = shift_wavelength(wav_cent, v_shift * (units.km / units.s))
            new_flux = interp1d(new_wav, spec, fill_value='extrapolate')(
                wav_cent_mask
            )
            data.loc[(hdf_name), :] = new_flux

    renorm_ref = interp1d(wav_cent_mask, data)(renorm_wavelength)
    norm = data.divide(renorm_ref, axis=0)
    
    #v = norm.index.get_level_values('v')   # This is in the order of files, which is not the same
                                           # as the datatable

    #v = np.full(len(inds),7794)  # I am just hoping that it is calculating in km/s...
    ts = np.array(ts)

    pca = Pipeline(steps=[
        ('scaler', MinMaxScaler(feature_range=(-0.5, 0.5))),
        # ('scaler', FunctionTransformer(lambda x: x)),
        ('pca', PCA(n_components=n_components))
    ])
    pca.fit(norm.values)
    eigenvalues = pca.fit_transform(norm.values)

    #n1 = [phys_pms.iloc[i-1]['n1'] for i in inds]   # -1 because of the naming scheme
    #T_inner = [phys_pms.iloc[i-1]['T_inner'] for i in inds]
    #tau_ths = [phys_pms.iloc[i-1]['tau_th'] for i in inds]
    setup_dir = '/afs/mpa/home/csogeza/BrokenPSim/SimFiles/StoreDensityProfiles/'
    n1 = []
    for i in inds:
        with open(setup_dir+'red_scaled_model_5d_'+str(i)+'.yml') as f:
            yob = yaml.load(f)
        n1.append(round(float((yob["model"]["structure"]["filename"]).split('/')[-1].split('_')[1].split('-')[0]),3))

    #params_dict = {'v': v, 'T': T_inner, 'T_ph': T_phs, 'n1': n1, 'tau_th': tau_ths}
    #params_dict = {'T': T_inner, 'T_ph': T_phs, 'n1': n1, 'tau_th': tau_ths}
    params_dict = {'T': Ts, 'T_ph': T_phs, 'n1': n1, 'v_ph': v_phs} 
    # The T will be dropped anyways at the first step 

    params_frame = pd.DataFrame.from_dict(params_dict)
    mags = calculate_grid_magnitudes(files)
    print('Warning: Correction turned off')
    #correction = 5. * np.log10(v * ts * (units.day.to(units.s)) / 10. * (    # Factoring out the change in the luminosity
    #        units.km.to(units.pc)))
    #mags = mags.add(correction, axis=0)
    
    return PCA_Decomp(
            data=data, data_norm=norm, wav_cent=wav_cent_mask,
            pca=pca, eigenvalues=eigenvalues, data_orig=data_orig,
            mags=mags,
            params_frame=params_frame, ts=ts,
    )

#-----------------------------------------------------------------------------------

def get_abundances(hdf_name, abunds, ref_path):
    hdf_base = os.path.split(hdf_name)[-1]
    yml_name = hdf_base.split('.hdf')[0] + '.yml'
    yml_path = os.path.join(ref_path, yml_name)
    X1 = OrderedDict()

    with open(yml_path, mode='r') as f:
        list_doc = yaml.load(f)
        X = list_doc['model']['abundances']
        for abund in abunds:
            X1[abund] = float(X[abund])
    return X1


class T_v_emulator(object):
    def __init__(self, res, wav, shift=True, use_T_ph=True, a=(0.7, 1.3),
                 epm_epoch=None, normalize=True, v_shift=None,
                 telluric_mask='NaD', nu=1.5, nu_mag=1.5):  # nu_mag smoothness pm for the Matern kernel, might use 2.5 --> smoother curves
        self.params = res.params_frame  # res is a PCA_Decomp named tuple
        self.nu = nu  # Smoothness parameter for Matern kernel
        if use_T_ph:
            print "Use T_ph"
            to_drop = 'T'
        else:
            print "Use T_inner"
            to_drop = 'T_ph'
        self.params = self.params.drop(to_drop, axis='columns')
        if normalize:
            self.normalize = True
        else:
            self.normalize = False

        #self.git_hash = get_git_revision_hash()

        self._spec = None
        if epm_epoch:
            for field in epm_epoch._fields:
                if not field == 'spec':
                    setattr(self, field, getattr(epm_epoch, field))
                else:
                    self._spec = epm_epoch.spec.dered_flux

        rbf_l = [250, 250, 250]

        self.X = self.params.values
        if normalize:
            self.scaler = MinMaxScaler()
            self.yscaler = MinMaxScaler()
            self.y = self.yscaler.fit_transform(res.eigenvalues)

            self.X_ = self.X.copy()
            self.X = self.scaler.fit_transform(self.X)
        else:
            self.y = res.eigenvalues

        self.a_min = a[0]
        self.a_max = a[1]

        self.hull1 = None
        self.hull2 = None
        self.hull3 = None

        self.n_preset = None
        self.v_shift = v_shift
        self.a_prior = UniformPrior(self.a_min, self.a_max, name='A')
        self.phys_prior = CompoundPrior.from_frame(self.params)
        self.prior = self.a_prior + self.phys_prior
        print self.prior

        self._observational_process = None
        self._covariance = None

        self.n_components = res.eigenvalues.shape[1]
        self.eigenvalues = res.eigenvalues
        self.pca = res.pca.named_steps['pca']
        self.pca_scaler = res.pca.named_steps['scaler']
        self.spca = res.pca
        self.model_mags = res.mags
        self.wav_cent = res.wav_cent
        self.interpolators = []
        self.interpolators_df = {}
        self.wav = wav
        self.initialize_telluric_mask(telluric_mask)
        self.wav = self.wav[self.tmask]
        if self.spec is not None:
            self.spec = self.spec[self.tmask]
        self.shift = shift
        self.ndim = self.X.shape[1]
        if not self.normalize:
            kernel = C(1.0, (1e-3, 1e3)) * (RBF(rbf_l, (1e-1, 1e8))) + \
                WhiteKernel(1e-2, (1e-25, 100))
        else:
            rbf_l = [1.0] * self.ndim
            if not nu == 'rbf':
                gp_kern = Matern(rbf_l, (1e-3, 1e6), nu=self.nu)
            else:
                gp_kern = RBF(rbf_l, (1e-3, 1e6))
            kernel = C(1.0, (1e-5, 1e2)) * \
                gp_kern + \
                WhiteKernel(1e-2, (1e-30, 100))
        for i in range(self.n_components):
            print 'Fitting PCA component: {}'.format(i)
            gp1 = GaussianProcessRegressor(kernel=kernel,
                                           n_restarts_optimizer=10)
            # , normalize_y=True)
            gp1.fit(self.X, self.y[:, i])

            self.interpolators.append(gp1)

        self.model_mags_interpolators = {'B': {}, 'V': {}, 'I': {}}

        kernelmag = C(1, (1e-5, 1e4)) * Matern(rbf_l, (1e-3, 1e4), nu=nu_mag) + \
            WhiteKernel(1e-2, (1e-30, 10))
        for band in self.model_mags.columns:
            mags = self.model_mags[band].values
            gpmag = GaussianProcessRegressor(
                kernel=kernelmag, n_restarts_optimizer=10
            )
            print 'Fitting mags {}'.format(band)
            #  gpmag = Pipeline([('scale', MinMaxScaler()), ('gp', gpmag)])
            #gpmag = Pipeline([
            #    ('scale', StandardScaler()), ('gp', gpmag)]
            #)
            gpmag.fit(self.X, mags)
            self.model_mags_interpolators[band] = gpmag

    def setup_hulls(self, Temp_t = 6700):
        T_set = self.params_frame['T_ph'].values
        v_set = self.params_frame['v_ph'].values

        points1 = np.vstack((T_set[T_set < Temp_t], v_set[T_set < Temp_t])).T
        self.hull1 = ConvexHull(points1)

        points2 = np.vstack((T_set[T_set > Temp_t], v_set[T_set > Temp_t])).T
        self.hull2 = ConvexHull(points2)

        cond3 = (T_set > Temp_t - 250) & (T_set < Temp_t + 250)
        points3 = np.vstack((T_set[cond3],v_set[cond3])).T
        self.hull3 = ConvexHull(points3)

    def emulate(self, x, return_std=False, sample=False, prepare_cov=False, fixed_vph=False):
        i = -1
        eigen = np.zeros(self.pca.n_components)
        stds = np.zeros_like(eigen)
        
        # v1 = np.exp(x['v'])
        # v1 = x['v_ph']
        if fixed_vph:
            x.drop('v_ph', inplace=True)
        #    print(x)
        # x = x_inp.values

        if self.v_shift is not None:
            v1 = self.v_shift

        if self.normalize:
            x = self.scaler.transform(x)
        x = x.reshape(1, -1)

        wav_em = self.wav_cent
        if self.shift:
            c = const.c.to(units.km/units.s).value
            wav_em = self.wav_cent / np.sqrt((1 + v1 / c)/(1 - v1 / c))
        for interpolator1 in self.interpolators:
            i += 1
            if return_std:
                inter_value, std = interpolator1.predict(
                    x, return_std=return_std
                )
                inter_value, std = inter_value[0], std[0]
                stds[i] = std
            else:
                inter_value = interpolator1.predict(x,
                                                    return_std=return_std)[0]
                if sample:
                    inter_value = interpolator1.sample_y(
                        x, random_state=RandomState())[0][0]
            eigen[i] = inter_value

        if self.normalize:
            eigen = self.yscaler.inverse_transform(eigen)
            if return_std:
                stds *= self.yscaler.data_max_ - self.yscaler.data_min_
        emu_spec = self.spca.inverse_transform(eigen)
        if return_std:
            std_spec = ((self.pca.components_ *
                         stds.reshape(-1, 1))**2.).sum(axis=0)
            std_spec *= (self.pca_scaler.data_max_ -
                         self.pca_scaler.data_min_)**2.
            std_spec = np.sqrt(std_spec)

        emu_spec = interp1d(wav_em, emu_spec,
                            fill_value='extrapolates')(self.wav)
        emu_spec = emu_spec.flatten()
        if return_std:
            std_spec = interp1d(wav_em, std_spec,
                                fill_value='extrapolate')(self.wav)
            return emu_spec, std_spec.flatten()
        else:
            return emu_spec
        
    def loglike_corr(self, x):
        x_series = pd.Series(x, self.prior.names)
        if self.shift:
            x_pos = x_series[['T_ph','v_ph']].values
			bool_list = [in_hull(x_pos, t_hull) for t_hull in 
                            [self.hull1, self.hull2, self.hull3]]
            if 1 not in bool_list:
                return 9999
        if self.n_preset is not None:
            x_series['n1'] = self.n_preset
        x_emu = x_series.loc[self.params.columns]
        spec_emu = self.emulate(x_emu)
        #spec_emu = self.observational_process.transform(
        #    self.wav.values, spec_emu, x_series
        #)
        diff = (self.spec / self.spec.max() - x[0] *
                spec_emu / spec_emu.max())

        lnlike = self.covariance.compute_loglike_for_params(
            params=x_series, y=diff
        )
        return lnlike

    def neg_loglike_corr(self, x):
        return -self.loglike_corr(x)

    def do_max_like_fit(self, mode = 'DE'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if mode == 'Brute':
                res = brute(self.neg_loglike_corr, self.bounds, Ns=15, finish=None)
                self.max_like = pd.Series(res, index=self.prior.names)
            elif mode == 'Nelder-Mead':
                res = minimize(self.neg_loglike_corr, np.average(self.bounds, axis=1), 
                            method='Nelder-Mead', options={"maxiter": 100})
                self.max_like = pd.Series(res.x, index=self.prior.names)
            else:
                res = differential_evolution(self.neg_loglike_corr, self.bounds)
                self.max_like = pd.Series(res.x, index=self.prior.names)
    
        spec_emu = self.emulate(self.max_like.loc[self.params.columns])
    
        return MaxLikeResult_tr(
            wav = self.wav,
            spec = self.spec,
            max_like_spec = spec_emu,
            params=self.max_like
        )
		

#    def sample_corr(self, **kwargs):
#        with warnings.catch_warnings():
#            warnings.simplefilter("ignore")
#
#            print self.prior
#            ndim = self.prior.ndim
#
#            MySolver = type('MySolver', (Solver,),
#                            {'Prior': self.prior,
#                            'LogLikelihood': self.loglike_corr})
#
#            with tempfile.TemporaryDirectory(
#                    dir='/afs/mpa/temp/cvogl') as tempdir:
#                output_basename = tempdir + '/'
#                self.result = MySolver(
#                    n_dims=ndim, verbose=True,
#                    outputfiles_basename=output_basename,
#                    evidence_tolerance=0.5  # 2.5
#                )
#            self.samples = pd.DataFrame(self.result.samples,
#                                        columns=self.prior.names)

    def maximum_likelihood(self, log=False):
        bounds = self.initialize_bounds_from_prior()
        if log:
            self.callback = DiffEvoCallBack(columns=self.prior.names)
        else:
            self.callback = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = differential_evolution(self.neg_loglike_corr,
                                         self.bounds,
                                         callback=self.callback)

        self.max_like = pd.Series(res.x, index=self.prior.names)
        spec_emu = self.emulate(
            self.max_like.loc[self.params.columns]
        )
        spec_emu = self.observational_process.transform(
            self.wav.values, spec_emu, self.max_like
        )

        A = self.max_like['A']
        self.max_like_spec = A * spec_emu / spec_emu.max()
        if hasattr(self, 'mags'):
            self.calculate_theta()
            self.d_theta_vs_d_RV = self.calculate_d_theta_vs_d_RV()
        else:
            self.thetas, self.theta_vs = None, None
            self.d_theta_vs_d_RV = None

        return MaxLikeResult(
            wav=self.wav,
            spec=self.spec,
            max_like_spec=self.max_like_spec,
            params=self.max_like,
            thetas=self.thetas,
            theta_vs=self.theta_vs,
            d_theta_vs_d_RV=self.d_theta_vs_d_RV
        )

    def maximum_likelihood_multiple_E_BV(self, delta_E_BVs=[-0.01, 0.01]):
        E_BV = self.reddening.E_BV
        delta_E_BVs += [0.0]
        ml_results = {}
        for delta in delta_E_BVs:
            self.reddening.E_BV = E_BV + delta
            print("E_BV: {}, delta: {}, self.E_BV: {}".format(E_BV, delta,
                                                              self.reddening.E_BV))
            ml = self.maximum_likelihood()
            ml_results[self.reddening.E_BV] = ml
        self.reddening.E_BV = E_BV
        return ml_results

    # TODO: Make this work for samples
    def calculate_theta(self, max_like=True):
        if not hasattr(self, 'max_like') or not hasattr(self, 'mags'):
            print "No magnitudes supplied or no fit performed!"
            raise ValueError
        else:
            self.thetas = {}
            self.theta_vs = {}
            if max_like:
                v = self.max_like['v'] * (units.km / units.s)
                if 'ts' in self.max_like.keys():
                    ts = self.max_like['ts']
                else:
                    ts = self.ts
            else:
                v = self.samples['v'].values * (units.km / units.s)
            correction = 5. * np.log10(v.value * ts * (units.day.to(units.s)) / 10. * (
                units.km.to(units.pc)))
            mags_interp = self.model_mags_interpolators
            for band_comb in ['BV', 'BVI', 'VI']:
                delta_mags = np.zeros(len(band_comb))
                for i, band in enumerate(band_comb):
                    A_band = CCM_AS_AV_ratio[band] * self.A_V
                    if max_like:
                        X = self.X_ml
                    else:
                        X = self.X_samples

                    delta_mags[i] += mags_interp[band].predict(X)[0] + A_band
                    delta_mags[i] -= self.mags[band]
                res = differential_evolution(self.epsilon, bounds=[(-50., 50)],
                                             args=(delta_mags,))
                theta = 10.**res.x[0]
                # theta = 10.**(norm**-1 * (mags_m - mags_obs))
                theta = units.Quantity(theta).to(1e8 * units.km / units.Mpc)
                self.thetas[band_comb] = theta
                self.theta_vs[band_comb] = (theta / v).to(
                    units.day / units.Mpc
                )

        self.thetas = pd.Series(self.thetas)
        self.theta_vs = pd.Series(self.theta_vs)

    # TODO: make this work for samples
    def calculate_d_theta_vs_d_RV(self, delta=0.005):
        R_V = self.reddening.R_V
        self.reddening.R_V = R_V + delta
        self.calculate_theta()
        tv1 = self.theta_vs

        self.reddening.R_V = R_V - delta
        self.calculate_theta()
        tv2 = self.theta_vs

        self.reddening.R_V = R_V
        return (tv1 - tv2) / (2 * delta)

    def calculate_d_theta_vs_d_mag(self, delta=0.005):
        mags = self.mags.copy()
        d_theta_vs_d_mag = {}
        for band in self.mags.keys():
            self.mags[band] = mags[band] + delta
            self.calculate_theta()
            tv1 = self.theta_vs

            self.mags[band] = mags[band] - delta
            self.calculate_theta()
            tv2 = self.theta_vs
            d_theta_vs_d_mag[band] = (tv1 - tv2) / (2 * delta)

            self.mags[band] = mags[band]

        return pd.DataFrame(d_theta_vs_d_mag)

    @staticmethod
    def epsilon(logtheta, delta_mags):
        """Squared difference of observed and model mags for logtheta"""
        return ((delta_mags - 5. * logtheta)**2.).sum()

    def random_fits(self, number=100, residuals=False, **kwargs):
        random_index = np.random.choice(
            np.arange(self.result.samples.shape[0]),
            size=number, replace=False
        )
        randomfits = np.zeros((number, self.tmask.sum()))
        spec = self.spec
        spec /= spec.max()
        for i, index in enumerate(random_index):
            x = self.result.samples[index, :]
            x_series = pd.Series(x, self.prior.names)
            x_emu = x_series.loc[self.params.columns]
            spec_emu = self.emulate(x_emu)
            spec_emu = self.observational_process.transform(
                self.wav.values, spec_emu, x_series
            )

            sample_spec = x[0] * spec_emu / spec_emu.max()
            if residuals:
                sample_spec -= spec
            randomfits[i] = sample_spec
            # TODO: This seems like it does not need to be in loop
            random_params = pd.DataFrame(
                self.result.samples[random_index, :],
                columns=self.prior.names
            )
        return randomfits, random_params

    @property
    def spec(self):
        return self._spec

    @property
    def observational_process(self):
        return self._observational_process

    @observational_process.setter
    def observational_process(self, observational_process):
        if not self._observational_process:
            self._observational_process = observational_process
            if observational_process.prior:
                self.prior += observational_process.prior
        else:
            raise Exception('Observational process already set')

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        if not self._covariance:
            self._covariance = covariance
            if covariance.prior:
                self.prior += covariance.prior
        else:
            raise Exception('Covariance already set')

    @spec.setter
    def spec(self, spec):
        self._spec = spec[self.tmask]

    def initialize_bounds_from_prior(self):
        bounds = []
        for prior in self.prior.priors:
            if type(prior) is UniformPrior:
                bounds.append((prior.pmin, prior.pmax))
            else:
                raise ValueError(
                    'Bounds can only be created from UniformPrior'
                )
        self.bounds = bounds

    def initialize_telluric_mask(self, tmask_type):
        telluric_mask = np.ones_like(self.wav, dtype='bool')
        for band in telluric_bands[tmask_type]:
            band_mask = np.logical_and(self.wav > band[0],
                                       self.wav < band[1])
            inverse_band_mask = np.logical_not(band_mask)
            telluric_mask = np.logical_and(inverse_band_mask, telluric_mask)
        self.tmask = telluric_mask

    @property
    def X_samples(self):
        phys_samples = self.samples[self.params.columns].values
        X_samples = self.scaler.transform(phys_samples)
        return X_samples

    @property
    def X_ml(self):
        phys_samples = self.max_like[self.params.columns].values
        X_ml = self.scaler.transform(phys_samples)
        return X_ml

    @property
    def A_V(self):
        A_V = 1.
        for key in ['E_BV', 'R_V']:
            if hasattr(self, 'samples'):
                if key in self.samples.columns:
                    value = self.samples[key].values
                else:
                    value = getattr(self.reddening, key)
            elif hasattr(self, 'max_like'):
                if key in self.max_like.index:
                    value = self.max_like[key]
                else:
                    value = getattr(self.reddening, key)
            A_V *= value
        return A_V

    @property
    def reddening(self):
        reddening = None
        for component in self.observational_process.components:
            if type(component) is Reddening:
                reddening = component
        return reddening
