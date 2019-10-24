"""Defines the `SFFCorrector` class.

`SFFCorrector` enables systematics to be removed from light curves using the
Self Flat-Fielding (SFF) method described in Vanderburg and Johnson (2014).

TODO
----
* Are we sufficiently compatible wiht the old SFFCorrector?
* Make robust if centroid values have NaNs.
"""
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

from patsy import dmatrix

from . import DesignMatrix, DesignMatrixCollection
from .regressioncorrector import RegressionCorrector
from ..utils import LightkurveWarning

from .. import MPLSTYLE

log = logging.getLogger(__name__)

__all__ = ['SFFCorrector']


class SFFCorrector(RegressionCorrector):
    """Special case of `.RegressionCorrector` where the `.DesignMatrix` includes
    the target's centroid positions.

    The design matrix also contains columns representing a spline in time
    design to capture the intrinsic, long-term variability of the target.

    Parameters
    ----------
    lc : `.LightCurve`
        The light curve that needs to be corrected.
    """
    def __init__(self, lc):
        # Setting these values as None so we don't get a value error if the
        # user calls before "correct()"

        self.raw_lc = lc
        if lc.flux_unit.to_string() == '':
            lc = lc.copy()
        else:
            lc = lc.copy().normalize()
        self.window_points = None
        self.windows = None
        self.bins = None
        self.timescale = None
        self.breakindex = None
        self.centroid_col = None
        self.centroid_row = None
        super(SFFCorrector, self).__init__(lc=lc)

    def __repr__(self):
        return 'SFFCorrector (LC: {})'.format(self.lc.targetid)

    def correct(self, centroid_col=None, centroid_row=None, windows=20, bins=5,
                timescale=1.5, breakindex=None, degree=3, restore_trend=False,
                additional_design_matrix=None, **kwargs):
        """Find the best fit correction for the light curve.

        Parameters
        ----------
        centroid_col : np.ndarray of floats (optional)
            Array of centroid column positions. If ``None``, will use the
            `centroid_col` attribute of the input light curve by default.
        centroid_row : np.ndarray of floats (optional)
            Array of centroid row positions. If ``None``, will use the
            `centroid_row` attribute of the input light curve by default.
        windows : int
            Number of windows to split the data into to perform the correction.
            Default 20.
        bins : int
            Number of "knots" to place on the arclength spline. More bins will
            increase the number of knots, making the spline smoother in arclength.
            Default 10.
        timescale: float
            Time scale of the b-spline fit to the light curve in time, in units
            of input light curve time.
        breakindex : None, int or list of ints (optional)
            Optionally the user can break the light curve into sections. Set
            break index to either an index at which to break, or list of indicies.
        degree : int
            The degree of polynomials in the splines in time and arclength. Higher
            values will create smoother splines. Default 3.
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        sigma : int (default 5)
            Standard deviation at which to remove outliers from fitting
        niters : int (default 5)
            Number of iterations to fit and remove outliers
        restore_trend : bool (default False)
            Whether to restore the long term spline trend to the light curve
        propagate_errors : bool (default False)
            Whether to propagate the uncertainties from the regression. Default is False.
            Setting to True will increase run time, but will sample from multivariate normal
            distribution of weights.
        additional_design_matrix : `~lightkurve.lightcurve.Correctors.DesignMatrix` (optional)
            Additional design matrix to remove, e.g. containing background vectors.

        Returns
        -------
        corrected_lc : `~lightkurve.lightcurve.LightCurve`
            Corrected light curve, with noise removed.
        """
        if centroid_col is None:
            centroid_col = self.lc.centroid_col
        if centroid_row is None:
            centroid_row = self.lc.centroid_row

        if np.any([~np.isfinite(centroid_row), ~np.isfinite(centroid_col)]):
            raise ValueError('Centroids contain NaN values.')

        self.window_points = _get_window_points(centroid_col, centroid_row, windows)
        self.windows = windows
        self.bins = bins
        self.timescale = timescale
        self.breakindex = breakindex

        # We make an approximation that the arclength is simply
        # (row**2 + col**2)**0.5
        # However to make this work row and column must be correlated not anticorrelated
        c = centroid_col - np.nanmin(centroid_col)
        r = centroid_row  - np.nanmin(centroid_row)
        # Force c to be correlated not anticorrelated
        if (np.polyfit(c, r, 1)[0] < 0):
            c = np.max(c) - c
        self.arclength = (c**2 + r**2)**0.5

        lower_idx = np.asarray(np.append(0, self.window_points), int)
        upper_idx = np.asarray(np.append(self.window_points, len(self.lc.time)), int)

        stack = []
        columns = []
        prior_sigmas = []
        for idx, a, b in zip(range(len(lower_idx)), lower_idx, upper_idx):
            knots = list(np.percentile(self.arclength[a:b], np.linspace(0, 100, bins+1)[1:-1]))
            ar = np.copy(self.arclength)
            ar[~np.in1d(ar, ar[a:b])] = 0
            dm = np.asarray(dmatrix("bs(x, knots={}, degree={}, include_intercept={}) - 1"
                                    "".format(knots, degree, True), {"x": ar}))
            stack.append(dm)
            columns.append(['window{}_bin{}'.format(idx+1, jdx+1)
                            for jdx in range(len(dm.T))])

            # I'm putting VERY weak priors on the SFF motion vectors
            prior_sigmas.append(np.ones(len(dm.T)) * 10000 * self.lc[a:b].flux.std())

        sff_dm = DesignMatrix(pd.DataFrame(np.hstack(stack)),
                              columns=np.hstack(columns),
                              name='sff',
                              prior_sigma=np.hstack(prior_sigmas))


        # long term
        n_knots = int((self.lc.time[-1] - self.lc.time[0])/timescale)
        s_dm = _get_spline_dm(self.lc.time, n_knots=n_knots, include_intercept=True)

        means = [np.average(self.lc.flux, weights=s_dm.values[:, idx]) for idx in range(s_dm.shape[1])]
        s_dm.prior_mu = np.asarray(means)

        # I'm putting WEAK priors on the spline that it must be around 1
        s_dm.prior_sigma = np.ones(len(s_dm.prior_mu)) * 1000 * self.lc.flux.std()



        # additional
        if additional_design_matrix is not None:
            if not isinstance(additional_design_matrix, DesignMatrix):
                raise ValueError('`additional_design_matrix` must be a DesignMatrix object.')
            self.additional_design_matrix = additional_design_matrix
            dm = DesignMatrixCollection([s_dm,
                                         sff_dm,
                                         additional_design_matrix])
        else:
            dm = DesignMatrixCollection([s_dm, sff_dm])

        # correct
        clc = super(SFFCorrector, self).correct(dm, **kwargs)

        # clean
        if restore_trend:
            trend = self.diagnostic_lightcurves['spline'].flux
            clc += trend - np.nanmedian(trend)
        clc *= self.raw_lc.flux.mean()

        return clc

    def diagnose(self):
        """Returns a diagnostic plot which visualizes what happened during the
        most recent call to `correct()`."""
        axs = self._diagnostic_plot()
        for t in self.window_points:
            axs[0].axvline(self.lc.time[t], color='r', ls='--', alpha=0.3)

    def diagnose_arclength(self):
        """Returns a diagnostic plot which visualizes arclength vs flux
        from most recent call to `correct()`."""

        max_plot = 5
        with plt.style.context(MPLSTYLE):
            fig, axs = plt.subplots(int(np.ceil(self.windows/max_plot)), max_plot,
                                    figsize=(10, int(np.ceil(self.windows/max_plot)*2)),
                                    sharex=True, sharey=True)
            axs = np.atleast_2d(axs)
            axs[0, 2].set_title('Arclength Plot/Window')
            plt.subplots_adjust(hspace=0, wspace=0)

            lower_idx = np.asarray(np.append(0, self.window_points), int)
            upper_idx = np.asarray(np.append(self.window_points, len(self.lc.time)), int)
            if hasattr(self, 'additional_design_matrix'):
                name = self.additional_design_matrix.name
                f = (self.lc.flux - self.diagnostic_lightcurves['spline'].flux
                            - self.diagnostic_lightcurves[name].flux)
            else:
                f = (self.lc.flux - self.diagnostic_lightcurves['spline'].flux)

            m = self.diagnostic_lightcurves['sff'].flux

            idx, jdx = 0, 0
            for a, b in zip(lower_idx, upper_idx):
                ax = axs[idx, jdx]
                if jdx == 0:
                    ax.set_ylabel('Flux')

                ax.scatter(self.arclength[a:b], f[a:b], s=1, label='Data')
                ax.scatter(self.arclength[a:b][~self.cadence_mask[a:b]], f[a:b][~self.cadence_mask[a:b]], s=10, marker='x', c='r', label='Outliers')

                s = np.argsort(self.arclength[a:b])
                ax.scatter(self.arclength[a:b][s], (m[a:b] - np.median(m[a:b]) + np.median(f[a:b]))[s], c='C2', s=0.5, label='Model')
                jdx += 1
                if jdx >= max_plot:
                    jdx = 0
                    idx += 1
                if b == len(self.lc.time):
                    ax.legend()


######################
#  Helper functions  #
######################

def _get_spline_dm(x, n_knots=20, degree=3, name='spline',
                   include_intercept=False):
    """Returns a `.DesignMatrix` which models splines using `patsy.dmatrix`.

    Parameters
    ----------
    x : np.ndarray
        vector to spline
    n_knots: int
        Number of knots (default: 20).
    degree: int
        Polynomial degree
    name: string
        Name to pass to `.DesignMatrix` (default: 'spline').
    include_intercept: bool
        Whether to include row of ones to find intercept. Default False.

    Returns
    -------
    dm: `.DesignMatrix`
        Design matrix object with shape (len(x), n_knots*degree).
    """
    dm_formula = "bs(x, df={}, degree={}, include_intercept={}) - 1" \
                 "".format(n_knots, degree, include_intercept)
    spline_dm = np.asarray(dmatrix(dm_formula, {"x": x}))
    df = pd.DataFrame(spline_dm, columns=['knot{}'.format(idx + 1)
                                          for idx in range(n_knots)])
    return DesignMatrix(df, name=name)


def _get_centroid_dm(col, row, name='centroids'):
    """Returns a `.DesignMatrix` containing (col, row) centroid positions
    and transformations thereof.

    Parameters
    ----------
    col : np.ndarray
        centroid column
    row : np.ndarray
        centroid row
    name : str
        Name to pass to `.DesignMatrix` (default: 'centroids').

    Returns
    -------
    dm: np.ndarray
        Design matrix with shape len(c) x 10
    """
    data = [col, row,
            col**2, row**2,
            col**3, row**3,
            col*row,
            col**2 * row, col * row**2,
            col**2 * row**2]
    names = [r'col', r'row',
             r'col^2', r'row^2',
             r'col^3', r'row^3',
             r'col \times row',
             r'col^2 \times row', r'col \times row^2',
             r'col^2 \times row^2']
    df = pd.DataFrame(np.asarray(data).T, columns=names)
    return DesignMatrix(df, name=name)


def _get_thruster_firings(arclength):
    """Find locations where K2 fired thrusters

    Parameters
    ----------
    arc : np.ndarray
        arclength as a function of time

    Returns
    -------
    thrusters: np.ndarray of bools
        True at times where thrusters were fired.
    """
    arc = np.copy(arclength)
    # Rate of change of rate of change of arclength wrt time
    d2adt2 = (np.gradient(np.gradient(arc)))
    # Fit a Gaussian, most points lie in a tight region, thruster firings are outliers
    g = models.Gaussian1D(amplitude=100, mean=0, stddev=0.01)
    fitter = fitting.LevMarLSQFitter()
    h = np.histogram(d2adt2[np.isfinite(d2adt2)], np.arange(-0.5, 0.5, 0.0001), density=True);
    xbins = h[1][1:] - np.median(np.diff(h[1]))
    g = fitter(g, xbins, h[0], weights=h[0]**0.5)

    # Depending on the orientation of the roll, it is hard to return
    # the point before the firing or the point after the firing.
    # This makes sure we always return the same value, no matter the roll orientation.
    def _start_and_end(start_or_end):
        """Find points at the start or end of a roll."""
        if start_or_end == 'start':
            thrusters = (d2adt2 < (g.stddev * -5)) & np.isfinite(d2adt2)
        if start_or_end == 'end':
            thrusters = (d2adt2 > (g.stddev * 5)) & np.isfinite(d2adt2)
        # Pick the best thruster in each cluster
        idx = np.array_split(np.arange(len(thrusters)), np.where(np.gradient(np.asarray(thrusters, int)) == 0)[0])
        m = np.array_split(thrusters, np.where(np.gradient(np.asarray(thrusters, int)) == 0)[0])
        th = []
        for jdx in range(len(idx)):
            if m[jdx].sum() == 0:
                th.append(m[jdx])
            else:
                th.append((np.abs(np.gradient(arc)[idx[jdx]]) == np.abs(np.gradient(arc)[idx[jdx]][m[jdx]]).max()) & m[jdx])
        thrusters = np.hstack(th)
        return thrusters

    # Get the start and end points
    thrusters = np.asarray([_start_and_end('start'), _start_and_end('end')])
    thrusters = thrusters.any(axis=0)

    # Take just the first point.
    thrusters = (np.gradient(np.asarray(thrusters, int)) >= 0) & thrusters
    return thrusters


def _get_window_points(centroid_col, centroid_row, windows, arclength=None, breakindex=None):
    """Returns indices where thrusters are fired.

    Parameters
    ----------
    lc : lk.LightCurve object
        Input light curve
    windows: int
        Number of windows to split the light curve into
    arc: np.ndarray
        Arclength for the roll motion
    breakindex: int
        Cadence where there is a natural break. Windows will be automatically put here.
    """
    if windows == 1:
        return []

    if arclength is None:
        # Compute arclength
        c = centroid_col - np.nanmin(centroid_col)
        r = centroid_row  - np.nanmin(centroid_row)
        if (np.polyfit(c, r, 1)[0] < 0):
            c = np.max(c) - c
        arclength = (c**2 + r**2)**0.5
#        arclength = ((c)**2 + (r)**2)**0.5

    # Validate break indicies
    if isinstance(breakindex, int):
        breakindexes = [breakindex]
    elif breakindex is None:
        breakindexes = []
    elif breakindex == 0:
        breakindexes = []
    else:
        breakindexes = breakindex
    if not isinstance(breakindexes, list):
        raise ValueError('`breakindex` must be an int or a list')

    # Find evenly spaced window points
    dt = len(centroid_col)/windows
    lower_idx = np.append(0, breakindexes)
    upper_idx = np.append(breakindexes, len(centroid_col))
    window_points = np.hstack([np.asarray(np.arange(a, b, dt), int) for a, b in zip(lower_idx, upper_idx)])

    # Get thruster firings
    thrusters = _get_thruster_firings(arclength)
    thrusters[breakindex] = True
    thrusters = np.where(thrusters)[0]

    # Find the nearest point to each thruster firing, unless it's a user supplied break point
    window_points = [thrusters[np.argmin(np.abs(wp - thrusters))] + 1
                         for wp in window_points
                         if wp not in breakindexes]
    window_points = np.unique(np.hstack([window_points, breakindexes]))

    # If the first window point is very short, just ignore that one
    if window_points[0] < (np.median(np.diff(window_points)) * 0.4):
        window_points = window_points[1:]
    return np.asarray(window_points, dtype=int)
