import scipy.interpolate
import numpy as np


class BackgroundIntegralDistribution(object):

    def __init__(self, fit_function_instance, best_fit_parameters, tstart, tstop, interpolating_degree=2, binsize=0.1):

        self._fit_function = fit_function_instance

        # First we need the integral distribution of the background
        bins = np.arange(tstart, tstop + 2, binsize)

        intxs = self._fit_function(bins[:-1], bins[1:], best_fit_parameters)

        interpolator = scipy.interpolate.InterpolatedUnivariateSpline((bins[1:] + bins[:-1]) / 2.0, np.cumsum(intxs),
                                                                      k=interpolating_degree)

        self._interpolator = interpolator

    def __call__(self, tt):

        return self._interpolator(tt)