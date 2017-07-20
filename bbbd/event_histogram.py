import numpy as np
import numexpr
import os
import scipy.optimize
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt


from bbbd.util.logging_system import get_logger
from bbbd.util.io_utils import sanitize_filename
from bbbd.instrument_specific.Fermi_LAT_LLE import LLEExposure
from bbbd.statistic.loglike import poisson_log_likelihood_no_bkg

# Get the logger
logger = get_logger(os.path.basename(__file__))


class EventHistogram(object):

    def __init__(self, event_times, bin_size, exposure_function, tstart, tstop, reference_time):

        # Store reference time
        self._reference_time = float(reference_time)

        # Make sure we have a numpy array and sort it

        self._times = np.array(event_times, ndmin=1, dtype=float) - reference_time

        self._times.sort()

        # Store scripts size making sure it is a float

        self._bin_size = float(bin_size)

        # Make the histogram

        # Compute the edges of the bins
        edges = np.arange(tstart - reference_time, tstop - reference_time, self._bin_size)

        # np.arange does not include the last element, so add it at the end
        edges = np.append(edges, [tstop - reference_time])

        # Get the histogram

        self._counts, _ = np.histogram(self._times, bins=edges)

        # Save bins start and stop
        self._edges = edges
        self._bin_starts = edges[:-1]
        self._bin_stops = edges[1:]
        self._bin_widths = self._bin_stops - self._bin_starts  # type: np.ndarray
        self._bin_centers = 0.5 * (self._bin_stops + self._bin_starts)  # type: np.ndarray

        # Now compute the exposure for each scripts
        if exposure_function is not None:

            self._exposure = np.array(map(lambda (t1,t2): exposure_function(t1 + reference_time, t2 + reference_time),
                                          zip(self._bin_starts, self._bin_stops)))

        else:

            # No exposure provided, just use the size of the bins
            logger.info("No exposure function provided, assuming 100% livetime")

            self._exposure = self._bin_widths

    @property
    def arrival_times(self):

        return self._times

    def plot(self, **kwargs):

        fig, (sub, sub2) = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[2, 1]})

        idx = self._exposure > 0

        rr = np.zeros_like(self._edges)
        rr[:-1][idx] = self._counts[idx] / self._exposure[idx]

        sub.step(self._edges, rr, where='post', **kwargs)

        sub.set_ylabel("Rate (cts/s)")

        sub2.plot(self._bin_centers, self._exposure)

        sub2.set_ylabel("Exposure (s)")
        sub2.set_xlabel("Time since %s" % self._reference_time)

        return fig

    @classmethod
    def from_ft1_and_ft2(cls, lle_or_ft1_file, pointing_file, selection=None, bin_size=1.0):
        """
        Build the function starting from a LLE (or standard LAT) data set.

        :param lle_or_ft1_file: file containing the events
        :param pointing_file: file containing the pointing and livetime information (aka FT2)
        :param selection: a string defininig a selection for the events, like "(ENERGY >= 30) & (ENERGY <= 100)".
        :param bin_size: (default: 1.0) binsize for the histogram
        :return: a EventHistogram instance
        """

        lle = LLEExposure(lle_or_ft1_file, pointing_file)

        instance = EventHistogram.from_events_fits(lle_or_ft1_file, lle.tstart, lle.tstop, lle.trigger_time,
                                                   selection=selection,
                                                   bin_size=bin_size,
                                                   time_column="TIME",
                                                   event_extension="EVENTS",
                                                   exposure_function=lle.get_exposure)

        instance.lle_exposure = lle

        return instance

    @classmethod
    def from_events_fits(cls, fits_filename, tstart, tstop, reference_time, selection=None,
                         bin_size=1.0, time_column="TIME", event_extension='EVENTS',
                         exposure_function=None):

        # Open FITS file
        with pyfits.open(sanitize_filename(fits_filename)) as f:

            # Get the extension data

            data = f[event_extension].data

        # Select in time
        idx = (data[time_column] >= tstart) & (data[time_column] <= tstop)
        data = data[idx]

        if selection is not None:

            # Select the events according to the selection

            try:

                idx = numexpr.evaluate(selection, local_dict=data)

            except IndexError:

                raise IndexError("Error in expression. You are probably using a column that does not exist. "
                                 "Mind that column names are case sensitive.")

            # Apply the selection
            data = data[idx]

        # Build the class

        return cls(data[time_column], bin_size=bin_size, tstart=tstart, tstop=tstop, reference_time=reference_time,
                   exposure_function=exposure_function)

    def fit_background(self, fit_polynomial, *intervals):

        mask = np.array(np.zeros_like(self._bin_starts), dtype=bool)

        # Select data to fit
        for (t1, t2) in intervals:

            # Select all bins between t1 and t2
            idx = (self._bin_starts >= t1) & (self._bin_stops < t2) & (self._exposure > 0)

            mask[idx] = True

        logger.info("Selected %i points" % np.sum(mask))

        # First we perform a robust unweighted least square optimization, to find a reasonable first approximation

        initial_approx = np.polyfit(self._bin_centers[mask],
                                    self._counts[mask] / self._exposure[mask],
                                    fit_polynomial.degree)

        logger.info("Robust unweighted least square returned these coefficients: %s" % (initial_approx))

        def _objective_function(coefficients):

            expected_rate = fit_polynomial(self._bin_starts[mask], self._bin_stops[mask], coefficients)

            expected_counts = expected_rate * self._exposure[mask]

            log_like = poisson_log_likelihood_no_bkg(self._counts[mask], expected_counts)

            return -log_like

        result = scipy.optimize.minimize(_objective_function,
                                         initial_approx,
                                         method='BFGS')

        logger.info("Fit results:")
        logger.info("Coefficients: %s" % map(lambda x:"%.3g" % x, result.x))
        logger.info("Likelihood value: %s" % result.fun)

        fig, sub = plt.subplots(1, 1)

        idx = self._exposure > 0
        rr = np.zeros_like(self._edges)
        rr[:-1][idx] = self._counts[idx] / self._exposure[idx]

        sub.step(self._edges, rr, where='post')

        rr2 = fit_polynomial(self._bin_starts, self._bin_stops, result.x)
        rr2 = np.append(rr2, 0)

        sub.step(self._edges,
                 rr2,
                 where='post')

        sub.set_xlabel("Time since %s" % self._reference_time)
        sub.set_ylabel("Rate (cts/s)")

        sub.set_xlim([self._bin_starts[mask].min(), self._bin_stops[mask].max()])
        sub.set_ylim([0.5 * rr.min(), 1.5 * rr.max()])

        return result.x, result.fun, fig
