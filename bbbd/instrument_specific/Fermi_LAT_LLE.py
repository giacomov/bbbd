import astropy.io.fits as pyfits
import numpy as np
import scipy.interpolate
import warnings

from bbbd.util.intervals import TimeInterval

class NoGTI(RuntimeError):
    pass


class LLEExposure(object):

    def __init__(self, lle_file, ft2_file):

        # Read GTIs and trigger time from FT1

        with pyfits.open(lle_file) as ft1_:

            self._tstart = ft1_['EVENTS'].header['TSTART']
            self._tstop = ft1_['EVENTS'].header['TSTOP']
            self._gti_start = ft1_['GTI'].data['START']
            self._gti_stop = ft1_['GTI'].data['STOP']
            self._trigger_time = ft1_['EVENTS'].header['TRIGTIME']

            # Make sure we have at least one event and a GTI
            if len(self._gti_start) == 0:

                raise NoGTI("No GTI in FT1 %s" % lle_file)

        # Read FT2 file

        with pyfits.open(ft2_file) as ft2_:

            ft2_tstart = ft2_['SC_DATA'].data.field("START")
            ft2_tstop = ft2_['SC_DATA'].data.field("STOP")
            ft2_livetime = ft2_['SC_DATA'].data.field("LIVETIME")

        ft2_bin_size = 1.0  # seconds

        if not np.all(ft2_livetime <= 1.0):

            warnings.warn("You are using a 30s FT2 file. You should use a 1s Ft2 file otherwise the livetime "
                          "correction will not be accurate!")

            ft2_bin_size = 30.0  # s

        # Keep only the needed entries (plus a padding of 10 bins)
        idx = (ft2_tstart >= self._gti_start.min() - 10 * ft2_bin_size) & \
              (ft2_tstop  <= self._gti_stop.max()  + 10 * ft2_bin_size)

        if np.sum(idx) == 0:

            raise NoGTI("No GTIs in file %s" % ft2_file)

        self._ft2_tstart = ft2_tstart[idx]
        self._ft2_tstop = ft2_tstop[idx]
        self._livetime = ft2_livetime[idx]

        # Now sort all vectors
        idx = np.argsort(self._ft2_tstart)

        self._ft2_tstart = self._ft2_tstart[idx]
        self._ft2_tstop = self._ft2_tstop[idx]
        self._livetime = self._livetime[idx]

        # Setup livetime computation
        self._livetime_interpolator = self._setup_livetime_computation()

    def get_exposure(self, t1, t2):
        """
        Returns the exposure between t1 and t2 (in MET) based on an interpolation of the livetime information
        contained in the ft2 file

        :param t1: start time in MET
        :type float
        :param t2: stop time in MET
        :type float
        :return: livetime between t1 and t2
        """

        # Make sure both t1 and t2 are within a GTI, otherwise returns zero exposure

        for tt in [t1, t2]:  # type: float

            try:

                _ = self._livetime_interpolator(tt - self._trigger_time)

            except ValueError:

                # t1 is outside of a GTI, return zero
                return 0

        return self._livetime_interpolator.integral(t1 - self._trigger_time, t2 - self._trigger_time)

    def _setup_livetime_computation(self):

        # These lists will contain the points for the interpolator

        xs = []
        ys = []

        # Pre-compute all time mid-points
        mid_points = (self._ft2_tstart + self._ft2_tstop) / 2.0  # type: np.ndarray

        # now loop through each GTI interval and setup the points for the livetime interpolator

        for start, stop in zip(self._gti_start, self._gti_stop):

            # create an index of all the FT2 bins falling within this interval

            tmp_idx = np.logical_and(start <= self._ft2_tstart, self._ft2_tstop <= stop)

            this_xs = mid_points[tmp_idx]
            this_dt = (self._ft2_tstop[tmp_idx] - self._ft2_tstart[tmp_idx])
            this_ys = self._livetime[tmp_idx] / this_dt

            # Now add one point at the beginning and one at the end with livetime exactly equal to the livetime
            # in that element, and then another point immediately before or after with livetime 0, so that the
            # interpolator will give you 0 if you use it within a bad time interval
            this_xs = np.insert(this_xs, [0, this_xs.shape[0]], [self._ft2_tstart[tmp_idx][0],
                                                                 self._ft2_tstop[tmp_idx][-1]])
            this_ys = np.insert(this_ys, [0, this_ys.shape[0]], [this_ys[0], this_ys[-1]])

            this_xs = np.insert(this_xs, [0, this_xs.shape[0]], [this_xs[0] - 1, this_xs[-1] + 1])
            this_ys = np.insert(this_ys, [0, this_ys.shape[0]], [0.0, 0.0])

            xs.extend(this_xs)
            ys.extend(this_ys)

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)

        # Note; ext=2 means that the interpolator will raise an error if an attempt is made to use it outside
        # of the provided range of values
        self._xs = xs
        self._ys = ys

        return scipy.interpolate.InterpolatedUnivariateSpline(xs - self._trigger_time,
                                                              ys,
                                                              w=np.ones_like(xs),
                                                              k=1,
                                                              check_finite=True,
                                                              ext=2)

    def is_in_gti(self, time):
        """

        Checks if a time falls within
        a GTI

        :param time: time in MET
        :return: bool
        """

        in_gti = False

        for start, stop in zip(self._gti_start, self._gti_stop):

            if (start <= time) and (time <= stop):

                in_gti = True

        return in_gti

    def is_interval_in_gti(self, t1, t2):
        """
        Check whether the provided interval is within a GTI, and returns a new interval reduced to the GTI

        :param t1:
        :param t2:
        :return:
        """

        requested_interval = TimeInterval(t1, t2)

        new_interval = None

        for start, stop in zip(self._gti_start, self._gti_stop):

            gti_interval = TimeInterval(start, stop)

            if gti_interval.overlaps_with(requested_interval):

                new_interval = gti_interval.intersect(requested_interval)

        if new_interval is None:

            return False, -1, -1

        else:

            return True, new_interval.start_time, new_interval.stop_time


    @property
    def tstart(self):
        """
        Access the start time of the file (the TSTART keyword in the header of the FT1 file)

        :return: tstart
        """
        return self._tstart

    @property
    def tstop(self):
        """
        Access the stop time of the file (the TSTOP keyword in the header of the FT1 file)

        :return: tstop
        """
        return self._tstop

    @property
    def trigger_time(self):
        return self._trigger_time