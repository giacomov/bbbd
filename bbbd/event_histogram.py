import numpy as np
import numexpr
import os
import scipy.optimize
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import copy

from bbbd.util.logging_system import get_logger
from bbbd.util.io_utils import sanitize_filename
from bbbd.instrument_specific.Fermi_LAT_LLE import LLEExposure
from bbbd.statistic.loglike import poisson_log_likelihood_no_bkg

# Get the logger
logger = get_logger(os.path.basename(__file__))


class NoDataToFit(RuntimeError):
    pass


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

        # This will contain the selection for the background fit
        self._background_mask = None

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

    def _get_cstat_distribution(self, fit_polynomial, coefficients, n_iteration, intervals):

        cstats = []

        best_fit_polys = []

        logger.info("Starting simulation of background for cstat distribution...")

        for i in range(n_iteration):

            if (i+1) % 100 == 0:

                logger.info("%.2f percent done" % (100.0 * (i+1) / float(n_iteration)))

            this_sim = self.get_simulation(fit_polynomial, coefficients, self._background_mask)

            try:

                best_fit_poly, cstat, _ = this_sim.fit_background(fit_polynomial, intervals, quiet=True, plot=False,
                                                                  background_mask=self._background_mask,
                                                                  initial_approx=coefficients)

            except:

                # Fit failed!
                logger.warn("Iteration %i failed" % (i+1))

                continue

            else:

                best_fit_polys.append(best_fit_poly)
                cstats.append(cstat)

        return np.array(cstats), np.array(best_fit_polys)

    def get_background_gof(self, best_fit_cstat, fit_polynomial, coefficients, n_iteration, intervals):

        cstats, best_fit_polys = self._get_cstat_distribution(fit_polynomial, coefficients, n_iteration, intervals)

        idx = (cstats >= best_fit_cstat)

        return float(np.sum(idx)) / cstats.shape[0], best_fit_polys

    def get_simulation(self, fit_polynomial, coefficients, mask):

        expected_rate = np.zeros_like(self._exposure)
        expected_rate[mask] = fit_polynomial(self._bin_starts[mask], self._bin_stops[mask], coefficients)

        randomized_counts = np.zeros_like(self._exposure)

        randomized_counts[mask] = np.random.poisson(expected_rate[mask] * self._exposure[mask])

        # Clone current histogram

        clone = copy.deepcopy(self)

        # Overwrite the counts with the normalized version
        clone._counts = randomized_counts

        # Remove the _times array so we are not tempted to use it
        clone._times = None

        return clone

    def fit_background(self, fit_polynomial, intervals, quiet=False, plot=True, background_mask=None,
                       initial_approx=None, theta_max=90):

        if background_mask is None:

            background_mask = np.array(np.zeros_like(self._bin_starts), dtype=bool)

            # Select data to fit
            for (t1, t2) in intervals:

                # Select all bins between t1 and t2
                idx = (self._bin_starts >= t1) & (self._bin_stops < t2) & (self._exposure > 0)

                background_mask[idx] = True

        if np.sum(background_mask) == 0:

            raise NoDataToFit("The background mask resulted in zero bins")

        # Compute the cos(theta) angle
        cos_theta = np.zeros_like(self._bin_centers, dtype=float)
        cos_theta[background_mask] = np.cos(fit_polynomial.theta_interpolator(self._bin_centers[background_mask]))

        # Remove from the background mask all the time intervals where the GRB is at more than 90 deg

        idx = cos_theta <= np.cos(np.deg2rad(theta_max))

        background_mask[idx] = False

        if not quiet: logger.info("Selected %i points" % np.sum(background_mask))

        # Store background mask
        self._background_mask = background_mask

        if initial_approx is None:

            # First we perform a robust unweighted least square optimization, to find a reasonable first approximation

            # The final fit function will be a polynomial multiplied by cos(theta). Since polyfit cannot
            # accept a user-defined function, we correct the data instead by diving them up by cos(theta)

            this_rate = self._counts[background_mask] / self._exposure[background_mask] / cos_theta[background_mask]

            this_counts_error = (1 + np.sqrt(self._counts[background_mask] + 0.75)) / cos_theta[background_mask]

            weight = 1 / (this_counts_error / self._exposure[background_mask]) # type: np.ndarray

            initial_approx = np.polyfit(self._bin_centers[background_mask],
                                        this_rate,
                                        fit_polynomial.degree,
                                        w=weight) # type: np.ndarray

            if not quiet:

                logger.info("Robust unweighted least square returned these coefficients: %s" % list(initial_approx))

        # Define the objective function (which is cstat)

        log_like_scale = 100.0

        def _objective_function(coefficients):

            expected_rate = fit_polynomial(self._bin_starts[background_mask], self._bin_stops[background_mask],
                                           coefficients)

            expected_counts = expected_rate * self._exposure[background_mask]

            log_like = poisson_log_likelihood_no_bkg(self._counts[background_mask], expected_counts)

            # Scale the log like to make easier the convergence

            return -log_like / log_like_scale

        if not quiet:

            logger.info("Starting value for statistic: %.3f" % (_objective_function(initial_approx) * log_like_scale))

        # Now scale the parameters (because SLSQP needs need to be more or less in the same ballpark, while
        # in general in a polynomial they are very different)

        scales = 10 ** (np.log10(np.abs(initial_approx)))

        fit_polynomial.set_scales(scales)

        # Scale the initial approximation

        initial_approx_scaled = initial_approx / scales

        # Construct bounds to avoid too wide variations
        bounds = []

        for a, b in zip(initial_approx_scaled / 100, initial_approx_scaled * 100):

            bounds.append(sorted([a, b]))

        # Minimize!

        result = scipy.optimize.minimize(_objective_function,
                                         initial_approx_scaled,
                                         options={'disp': False, 'ftol': 1e-2, 'maxiter': 100000},
                                         bounds=bounds,
                                         constraints=None)

        # Get the "true" best fit coefficients, i.e., the results of the fit multiplied by the scales

        best_fit_coefficients = result.x * scales

        # Remove scales
        fit_polynomial.remove_scales()

        if not quiet:
            logger.info("Fit results:")
            logger.info("Coefficients: %s" % map(lambda x:"%.3g" % x, best_fit_coefficients))
            logger.info("Likelihood value: %s" % (result.fun * log_like_scale))

        # Make sure the fit is good
        assert result.success == True, "Background fit failed!"

        if not np.all(fit_polynomial(self._bin_starts[background_mask], self._bin_stops[background_mask],
                                           result.x) >= 0):

            logger.warn("Polynomial is not positive everywhere.")

        # Always clip the polynomial at zero so it can never give negative fluxes (which would throw off simulations)

        fit_polynomial.clip_at_zero = True

        if plot:

            fig, sub = plt.subplots(1, 1)

            idx = self._exposure > 0
            rr = np.zeros_like(self._edges)
            rr[:-1][idx] = self._counts[idx] / self._exposure[idx]

            sub.step(self._edges, rr, where='post')

            rr2 = np.zeros_like(self._edges)
            # The polynomial can only be evaluated within the off-pulse extremes
            idx = ((self._bin_starts >= self._bin_starts[background_mask].min()) &
                   (self._bin_stops  <= self._bin_stops[background_mask].max()))

            rr2[:-1][idx] = fit_polynomial(self._bin_starts[idx],
                                           self._bin_stops[idx],
                                           best_fit_coefficients)

            sub.step(self._edges,
                     rr2,
                     where='post')

            sub.set_xlabel("Time since %s" % self._reference_time)
            sub.set_ylabel("Rate (cts/s)")

            sub.set_xlim([self._bin_starts[background_mask].min(), self._bin_stops[background_mask].max()])
            sub.set_ylim([0.5 * rr.min(), 1.5 * rr.max()])

        else:

            fig = None

        return best_fit_coefficients, result.fun, fig
