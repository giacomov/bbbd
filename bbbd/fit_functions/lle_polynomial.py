import numpy as np
import astropy.io.fits as pyfits
from astropy.coordinates.angle_utilities import angular_separation
import scipy.interpolate


class LLEPolynomial(object):

    def __init__(self, ra, dec, trigger_time, ft2file, degree=2):

        # Open the FT2 file and read the pointing information
        with pyfits.open(ft2file) as f:

            # Remember: in the FT2 all quantities (except the livetime) refer to the START time
            time = f['SC_DATA'].data['START']

            # Read the position of the Z-axis (the boresight) and convert it to radians
            ra_scz_rad = np.deg2rad(f['SC_DATA'].data['RA_SCZ'])
            dec_scz_rad = np.deg2rad(f['SC_DATA'].data['DEC_SCZ'])

        # Prepare the interpolator
        ang_dist_rad = angular_separation(ra_scz_rad, dec_scz_rad, np.deg2rad(ra), np.deg2rad(dec))

        self._theta_interpolator_radians = scipy.interpolate.InterpolatedUnivariateSpline(time - trigger_time,
                                                                                          ang_dist_rad,
                                                                                          w=np.ones_like(ang_dist_rad),
                                                                                          k=2,
                                                                                          ext=2,
                                                                                          check_finite=True)

        # Store the trigegr time as reference time
        self._reference_time = float(trigger_time)

        self._degree = int(degree)

        self._scales = np.ones(self._degree+1)

    @property
    def degree(self):
        return self._degree

    @property
    def reference_time(self):
        return self._reference_time

    def set_scales(self, scales):

        self._scales = scales

    def remove_scales(self):

        self._scales = np.ones(self._degree+1)

    def __call__(self, tstarts, tstops, poly_coefficients):

        t1 = tstarts
        t2 = tstops
        tt = 0.5 * (t2 + t1)

        this_poly = np.poly1d(poly_coefficients * self._scales)

        integral = this_poly.integ()

        integrals = (integral(t2) - integral(t1)) / (t2 - t1)

        thetas_rad = self._theta_interpolator_radians(tt)

        corr = np.cos(thetas_rad)

        expected_counts = integrals * corr

        # expected_counts = np.maximum(integrals * corr, 0)

        return expected_counts
