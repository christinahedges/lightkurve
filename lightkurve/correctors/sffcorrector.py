"""Defines SFFCorrector
"""
from __future__ import division, print_function

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import linalg, interpolate
from matplotlib import pyplot as plt

from astropy.stats import sigma_clip

from ..lightcurve import LightCurve
from .. import PACKAGEDIR


log = logging.getLogger(__name__)

__all__ = ['SFFCorrector']


class SFFCorrector(object):
    def __init__(self, lc, centroid_col=None, centroid_row=None, type='bspline'):
        self.lc = lc.remove_nans()
        self.flux = np.copy(self.lc.flux)
        self.flux_err = np.copy(self.lc.flux_err)
        self.time = np.copy(self.lc.time)
        self.model = np.ones(len(self.flux))
        self.type = type

        # Campaign break point in cadence number
        self.breakpoint = self._get_break_point(self.lc.campaign)
        # Campaign break point in index
        self.breakindex = np.argmin(np.abs(self.lc.cadenceno - self.breakpoint))
        # Input validation
        if centroid_col is None:
            try:
                self.centroid_col = self.lc.centroid_col
            except AttributeError:
                raise ValueError('`centroid_col` must be passed to `correct()` '
                                 'because it is not a property of the LightCurve.')
        else:
            self.centroid_col = centroid_col

        if centroid_row is None:
            try:
                self.centroid_row = self.lc.centroid_row
            except AttributeError:
                raise ValueError('`centroid_row` must be passed to `correct()` '
                                 'because it is not a property of the LightCurve.')
        else:
            self.centroid_row = centroid_row


    def __repr__(self):
        return 'SFFCorrector for object {}'.format(self.lc.label)

    def _get_break_point(self, campaign):
        ''' Find the breakpoint in a light curve based on the campaign. '''
        df = pd.read_csv('{}/correctors/data/breakpoints.txt'.format(PACKAGEDIR))
        loc = df['campaign'] == campaign
        if not loc.any():
            raise ValueError("No breakpoint for this campaign, Christina update this.")
        return int(np.asarray(df[loc]['breakpoint'])[0])

    def _fit_long_term_trends(self, outlier_mask=None):
        ''' Returns a model of the long term trends. '''
        # Flatten time
        if outlier_mask is None:
            nb = int(timescale/np.median(np.diff(self.time)))
            nb = [nb if nb % 2 == 1 else nb + 1][0]
            outlier_mask = ~(LightCurve(self.time, self.flux/self.model).flatten(nb).remove_outliers(sigma, return_mask=True)[1])

        if self.type is 'bspline':
            long_term_model = self._fit_bspline(outlier_mask=outlier_mask)
        if self.type is 'lomb-scargle':
            raise NotImplementedError
        if self.type is 'gp':
            raise NotImplementedError

        long_term_trend = long_term_model(self.time)
        long_term_trend /= np.median(long_term_trend)
        return  long_term_trend

    def _fit_bspline(self, outlier_mask):
        """Returns a `scipy.interpolate.BSpline` object to interpolate flux as a function of time."""
        '''Take in light curve, remove trend, pass back corrected light curve'''

        knots = np.arange(self.time[0], self.time[-1], self.timescale)
        # If the light curve has breaks larger than the spacing between knots,
        # we must remove the knots that fall in the breaks.
        # This is necessary for e.g. K2 Campaigns 0 and 10.
        bad_knots = []
        a = self.time[:-1][np.diff(self.time) > self.timescale]  # times marking the start of a gap
        b = self.time[1:][np.diff(self.time) > self.timescale]  # times marking the end of a gap
        for a1, b1 in zip(a, b):
            bad = np.where((knots > a1) & (knots < b1))[0][1:-1]
            [bad_knots.append(b) for b in bad]
        good_knots = list(set(list(np.arange(len(knots)))) - set(bad_knots))
        knots = knots[good_knots]

        # Now fit and return the spline
        t, c, k = interpolate.splrep(self.time[outlier_mask], (self.flux/self.model)[outlier_mask], t=knots[1:])
        return interpolate.BSpline(t, c, k)



    def _polynomial_correction(self, long_term_trend, mask, outlier_mask):
        build_components = lambda X, Y, T: np.array([X**4, X**3, X**2, X,
                                                     Y**4, Y**3, Y**2, Y,
                                                     X**4*Y**3, X**4*Y**2, X**4*Y, X**3*Y**2, X**3*Y, X**2*Y, X*Y,
                                                     Y**4*X**3, Y**4*X**2, Y**4*X, Y**3*X**2, Y**3*X, Y**2*X, np.ones(len(T))]).T
        components = build_components(self.centroid_col[mask], self.centroid_row[mask], self.time[mask])
        A = np.dot(components[outlier_mask[mask]].T, (components[outlier_mask[mask]]/(self.flux_err[mask & outlier_mask]**2)[:, None]))
        A[np.diag_indices_from(A)] += self.flux.mean() * 1e-6
        B = np.dot(components[outlier_mask[mask]].T, (((self.flux/self.model)/long_term_trend)[mask & outlier_mask]/(self.flux_err[mask & outlier_mask]**2))[:, None])
        C = np.linalg.solve(A, B)
        trend = np.dot(components, C).T[0]
        trend -= np.nanmedian(trend[trend > np.nanpercentile(trend, 75)])
        trend += np.nanmedian(self.flux)
        trend /= np.nanmedian(self.flux)

        return trend


    def _arclength_correction(self, long_term_trend, mask, outlier_mask):
        dcol = self.centroid_col[mask] - np.min(self.centroid_row[mask][outlier_mask[mask]])
        drow = self.centroid_row[mask] - np.min(self.centroid_row[mask][outlier_mask[mask]])
        arc = (dcol**2 + drow**2)**0.5
        s = np.argsort(arc[outlier_mask[mask]])


        norm_flux = (self.flux/self.model)[mask]/long_term_trend[mask]
        norm_flux /= np.median(norm_flux)
        arc1, norm_flux1 = arc[outlier_mask[mask]][s], norm_flux[outlier_mask[mask]][s]
        knots = np.array([np.median(split) for split in np.array_split(arc1, self.bins)])
        arc_model = interpolate.BSpline(*interpolate.splrep(arc1, norm_flux1, t=knots))
        arc_trend = arc_model(arc)/np.median(arc_model(arc[outlier_mask[mask]]))
        return arc_trend


    def _correction(self, window_points, type, outlier_mask, niters=3):
        """Performs a very basic correction to remove the worst motion

        Uses a 2D, 4th order polynomial fit to the column and row centroids in each
        half of the light curve. """

        for count in range(niters):
            long_term_trend = self._fit_long_term_trends(outlier_mask=outlier_mask)

            for idx in np.array_split(np.arange(len(self.time)), window_points):
                mask = np.in1d(np.arange(len(self.time)), idx)
                if type == 'polynomial':
                    self.model[mask] *= self._polynomial_correction(long_term_trend=long_term_trend, mask=mask, outlier_mask=outlier_mask)
                if type == 'arclength':
                    self.model[mask] *= self._arclength_correction(long_term_trend=long_term_trend, mask=mask, outlier_mask=outlier_mask)

            # long_term_trend = self._fit_long_term_trends(outlier_mask=outlier_mask)
            # ax = ((self.lc/self.model)/long_term_trend).plot()

            # k = self.model > np.percentile(self.model, 70)
            # l = np.polyval(np.polyfit(self.time[k], self.model[k], 1), self.time)
            # l /= np.median(l)
            # self.model /= l
        return



    def correct(self, cadence_mask=None, niters=3, windows=20, bins=10, restore_trend=True, timescale=1.5, sigma=5):
        # Renew the model
        self.model = np.ones(len(self.time))

        self.bins = bins
        self.windows = windows
        self.timescale = timescale
        self.sigma = sigma

        if cadence_mask is None:
            cadence_mask = np.ones(len(self.time), bool)

        self.window_points = np.append(np.linspace(0, self.breakindex + 1, self.windows//2 + 1, dtype=int)[1:],
                          np.linspace(self.breakindex + 1, len(self.time), self.windows//2 + 1, dtype=int)[1:-1])
        self.window_points[np.argmin((self.window_points - self.breakindex + 1)**2)] = self.breakindex + 1

        nb = int(self.timescale/np.median(np.diff(self.time)))
        nb = [nb if nb % 2 == 1 else nb + 1][0]


        self._correction(window_points=[self.breakindex], type='polynomial', outlier_mask=cadence_mask, niters=1)

        outlier_mask = np.copy(cadence_mask)
        outlier_mask &= ~(self.lc/self.model).flatten(nb).remove_outliers(self.sigma, return_mask=True)[1]

        self._correction(window_points=[self.breakindex], type='arclength', outlier_mask=outlier_mask, niters=1)

        outlier_mask = np.copy(cadence_mask)
        outlier_mask &= ~(self.lc/self.model).flatten(nb).remove_outliers(self.sigma, return_mask=True)[1]

        self._correction(window_points=self.window_points, type='arclength', outlier_mask=outlier_mask, niters=niters)

        if not restore_trend:
            outlier_mask = np.copy(cadence_mask)
            nb = int(self.timescale/np.median(np.diff(self.time)))
            nb = [nb if nb % 2 == 1 else nb + 1][0]
            outlier_mask &= ~(self.lc/self.model).flatten(nb).remove_outliers(self.sigma, return_mask=True)[1]
            long_term_trend = self._fit_long_term_trends(outlier_mask=outlier_mask)
            self.model *= long_term_trend

        return self.lc/self.model
