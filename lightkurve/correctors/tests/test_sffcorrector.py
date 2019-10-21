import pytest

import numpy as np
from numpy.testing import assert_almost_equal

from ... import LightCurve, KeplerLightCurveFile, KeplerLightCurve
from .. import SFFCorrector
from astropy.utils.data import get_pkg_data_filename


K2_C08 = ("https://archive.stsci.edu/missions/k2/lightcurves/c8/"
          "220100000/39000/ktwo220139473-c08_llc.fits")


@pytest.mark.remote_data
@pytest.mark.parametrize("path", [K2_C08])
def test_remote_data(path):
    """Can we correct a simple K2 light curve?"""
    lcf = KeplerLightCurveFile(path, quality_bitmask=None)
    sff = SFFCorrector(lcf.PDCSAP_FLUX.remove_nans())
    corrected_lc = sff.correct(windows=10, bins=5, timescale=0.5)


def test_sff_corrector():
    """Does our code agree with the example presented in Vanderburg
    and Johnson (2014)?"""
    # The following csv file, provided by Vanderburg and Johnson
    # at https://www.cfa.harvard.edu/~avanderb/k2/ep60021426.html,
    # contains the results of applying SFF to EPIC 60021426.
    fn = get_pkg_data_filename('../../tests/data/ep60021426alldiagnostics.csv')
    data = np.genfromtxt(fn, delimiter=',', skip_header=1)
    mask = data[:, -2] == 0  # indicates whether the thrusters were on or off
    time = data[:, 0]
    raw_flux = data[:, 1]
    corrected_flux = data[:, 2]
    centroid_col = data[:, 3]
    centroid_row = data[:, 4]
    arclength = data[:, 5]
    correction = data[:, 6]

    lc = LightCurve(time=time, flux=raw_flux, flux_err=np.ones(len(raw_flux)) * 0.0001)
    sff = SFFCorrector(lc)
    corrected_lc = sff.correct(centroid_col=centroid_col,
                               centroid_row=centroid_row, windows=1, restore_trend=True)
    assert (np.isclose(corrected_flux, corrected_lc.flux, atol=0.5e-3).all())

    # masking
    corrected_lc = sff.correct(centroid_col=centroid_col,
                               centroid_row=centroid_row, windows=1, restore_trend=True,
                               cadence_mask=mask)
    assert (np.isclose(corrected_flux, corrected_lc.flux, atol=0.5e-3).all())

    # masking and breakindex
    corrected_lc = sff.correct(centroid_col=centroid_col,
                               centroid_row=centroid_row, windows=1, restore_trend=True,
                               cadence_mask=mask, breakindex=150)
    assert (np.isclose(corrected_flux, corrected_lc.flux, atol=0.5e-3).all())

    # masking and breakindex and iters
    corrected_lc = sff.correct(centroid_col=centroid_col,
                               centroid_row=centroid_row, windows=1, restore_trend=True,
                               cadence_mask=mask, breakindex=150, niters=3)
    assert (np.isclose(corrected_flux, corrected_lc.flux, atol=0.5e-3).all())

    # masking and breakindex and bins
    corrected_lc = sff.correct(centroid_col=centroid_col,
                               centroid_row=centroid_row, windows=1, restore_trend=True,
                               cadence_mask=mask, breakindex=150, bins=5)
    assert (np.isclose(corrected_flux, corrected_lc.flux, atol=0.5e-3).all())


    # test using KeplerLightCurve interface
    klc = KeplerLightCurve(time=time, flux=raw_flux, flux_err=np.ones(len(raw_flux)) * 0.01, centroid_col=centroid_col,
                           centroid_row=centroid_row)
    sff = klc.to_corrector("sff")
    klc = sff.correct(windows=1, restore_trend=True)
    assert (np.isclose(corrected_flux, klc.flux, atol=0.5e-3).all())

    # Can plot
    sff.diagnose()

def test_sff_knots():
    """Is SFF robust against gaps in time and irregular time sampling?
    This test creates a random light curve with gaps in time between
    days 20-30 and days 78-80.  In addition, the time sampling rate changes
    in the interval between day 30 and 78.  SFF should fail without error.
    """
    n_points = 300
    fn = get_pkg_data_filename('../../tests/data/ep60021426alldiagnostics.csv')
    data = np.genfromtxt(fn, delimiter=',', skip_header=1)
    raw_flux = data[:, 1][:n_points]
    centroid_col = data[:, 3][:n_points]
    centroid_row = data[:, 4][:n_points]

    time = np.concatenate((np.linspace(0, 20, int(n_points/3)),
                           np.linspace(30, 78, int(n_points/3)),
                           np.linspace(80, 100, int(n_points/3))
                           ))
    lc = KeplerLightCurve(time=time,
                          flux=raw_flux,
                          flux_err=np.ones(n_points) * 0.0001,
                          centroid_col=centroid_col,
                          centroid_row=centroid_row)

    sff = SFFCorrector(lc)
    sff.correct()  # should not raise an exception
