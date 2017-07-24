# Author: Giacomo Vianello (giacomov@stanford.edu)

import logging
import time
import datetime
import multiprocessing
import numexpr
import numexpr.necompiler
import numpy as np
from functools import partial
import sys

from astropy.stats import bayesian_blocks as astropy_bb
from bbbd.util.logging_system import get_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bayesian_blocks")

# HACK: susbstitute the getArguments of numexpr to be faster in our particular case,
# where we always supply the local dictionary (see the bayesian block loop)
def getArguments(names, local_dict, *args, **kwargs):
    """Get the arguments based on the names."""

    return map(local_dict.__getitem__, names)
# Monkey patch numexpr
numexpr.necompiler.getArguments = getArguments


def simulation_worker(i, desired_p0, n_events):

    t = np.cumsum(np.random.exponential(1.0, size=n_events))

    idx = t <= n_events

    t = t[idx]

    # Since the rate is 1 and we start at zero, tstart=0 and tstop=n_events * 1=n_events

    blocks = bayesian_blocks(t, 0, n_events, desired_p0)

    return len(blocks) - 2


def calibrate_prior(desired_p0, n_events, n_sim=10000, n_cpu=multiprocessing.cpu_count()):

    logger = get_logger("calibrate_prior")

    # Generate n_events Poisson distributed with rate 1

    false_positive = 0

    partial_worker = partial(simulation_worker, desired_p0=desired_p0, n_events=n_events)

    start_time = time.time()

    if n_cpu > 0:

        pool = multiprocessing.Pool(processes=n_cpu)

        try:

            chunksize = 10

            for i, result in enumerate(pool.imap(partial_worker, range(n_sim), chunksize=chunksize)):

                false_positive += result

                if (i+1) % (chunksize * n_cpu) == 0:

                    this_time = time.time()
                    elapsed_time_seconds = this_time - start_time
                    elapsed_time_date = datetime.timedelta(seconds=elapsed_time_seconds)
                    remaining_time_seconds = (elapsed_time_seconds) / (i+1) * (n_sim - i - 1)
                    remaining_time_date = datetime.timedelta(seconds=remaining_time_seconds)

                    logger.info("%i out of %i completed" % (i+1, n_sim))
                    logger.info("Elapsed: %s, remaining: %s" % (elapsed_time_date, remaining_time_date))

        except:

            raise

        finally:

            pool.close()

    else:

        for i, result in enumerate(map(partial_worker, range(n_sim))):

            false_positive += result

            if i % 100 == 0:
                logger.info("%i out of %i completed" % (i + 1, n_sim))

    return float(false_positive) / n_sim


def bayesian_blocks_astropy(tt, ttstart, ttstop, p0, bkg_integral_distribution=None):
    """
    Divide a series of events characterized by their arrival time in blocks
    of perceptibly constant count rate. If the background integral distribution
    is given, divide the series in blocks where the difference with respect to
    the background is perceptibly constant.

    :param tt: arrival times of the events
    :param ttstart: the start of the interval
    :param ttstop: the stop of the interval
    :param p0: the false positive probability. This is used to decide the penalization on the likelihood, so this
    parameter affects the number of blocks
    :param bkg_integral_distribution: (default: None) If given, the algorithm account for the presence of the background and
    finds changes in rate with respect to the background
    :return: the np.array containing the edges of the blocks
    """

    # Verify that the input array is one-dimensional
    tt = np.asarray(tt, dtype=float)

    assert tt.ndim == 1

    if bkg_integral_distribution is not None:

        # Transforming the inhomogeneous Poisson process into an homogeneous one with rate 1,
        # by changing the time axis according to the background rate
        logger.debug("Transforming the inhomogeneous Poisson process to a homogeneous one with rate 1...")
        t = np.array(bkg_integral_distribution(tt))
        logger.debug("done")

        # Now compute the start and stop time in the new system
        tstart = bkg_integral_distribution(ttstart)
        tstop = bkg_integral_distribution(ttstop)

    else:

        t = tt
        tstart = ttstart
        tstop = ttstop

    # Create initial cell edges (Voronoi tessellation)
    edges = np.concatenate([[t[0]],
                            0.5 * (t[1:] + t[:-1]),
                            [t[-1]]])

    # Create the edges also in the original time system
    edges_ = np.concatenate([[tt[0]],
                             0.5 * (tt[1:] + tt[:-1]),
                             [tt[-1]]])


    # Create a lookup table to be able to transform back from the transformed system
    # to the original one
    lookup_table = {key: value for (key, value) in zip(edges, edges_)}

    edg = astropy_bb(t, fitness='events', p0=p0)

    # Transform the found edges back into the original time system

    if (bkg_integral_distribution is not None):

        final_edges = map(lambda x: lookup_table[x], edg)

    else:

        final_edges = edg

    return np.asarray(final_edges)



def bayesian_blocks(tt, ttstart, ttstop, p0, bkg_integral_distribution=None):
    """
    Divide a series of events characterized by their arrival time in blocks
    of perceptibly constant count rate. If the background integral distribution
    is given, divide the series in blocks where the difference with respect to
    the background is perceptibly constant.

    :param tt: arrival times of the events
    :param ttstart: the start of the interval
    :param ttstop: the stop of the interval
    :param p0: the false positive probability. This is used to decide the penalization on the likelihood, so this
    parameter affects the number of blocks
    :param bkg_integral_distribution: (default: None) If given, the algorithm account for the presence of the background and
    finds changes in rate with respect to the background
    :return: the np.array containing the edges of the blocks
    """

    # Verify that the input array is one-dimensional
    tt = np.asarray(tt, dtype=float)

    assert tt.ndim == 1

    if bkg_integral_distribution is not None:

        # Transforming the inhomogeneous Poisson process into an homogeneous one with rate 1,
        # by changing the time axis according to the background rate
        logger.debug("Transforming the inhomogeneous Poisson process to a homogeneous one with rate 1...")
        t = np.array(bkg_integral_distribution(tt))
        logger.debug("done")

        # Now compute the start and stop time in the new system
        tstart = bkg_integral_distribution(ttstart)
        tstop = bkg_integral_distribution(ttstop)

    else:

        t = tt
        tstart = ttstart
        tstop = ttstop

    # Create initial cell edges (Voronoi tessellation)
    edges = np.concatenate([[t[0]],
                            0.5 * (t[1:] + t[:-1]),
                            [t[-1]]])

    # Create the edges also in the original time system
    edges_ = np.concatenate([[tt[0]],
                             0.5 * (tt[1:] + tt[:-1]),
                             [tt[-1]]])


    # Create a lookup table to be able to transform back from the transformed system
    # to the original one
    lookup_table = {key: value for (key, value) in zip(edges, edges_)}

    # The last block length is 0 by definition
    block_length = tstop - edges

    if np.sum((block_length <= 0)) > 1:

        raise RuntimeError("Events appears to be out of order! Check for order, or duplicated events.")

    N = t.shape[0]

    # arrays to store the best configuration
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    # eq. 21 from Scargle 2012
    prior = 4 - np.log(73.53 * p0 * (N**-0.478))

    logger.debug("Finding blocks...")

    # This is where the computation happens. Following Scargle et al. 2012.
    # This loop has been optimized for speed:
    # * the expression for the fitness function has been rewritten to
    #  avoid multiple log computations, and to avoid power computations
    # * the use of scipy.weave and numexpr has been evaluated. The latter
    #  gives a big gain (~40%) if used for the fitness function. No other
    #  gain is obtained by using it anywhere else

    # Set numexpr precision to low (more than enough for us), which is
    # faster than high
    oldaccuracy = numexpr.set_vml_accuracy_mode('low')
    numexpr.set_num_threads(1)
    numexpr.set_vml_num_threads(1)

    # Speed tricks: resolve once for all the functions which will be used
    # in the loop
    numexpr_evaluate = numexpr.evaluate
    numexpr_re_evaluate = numexpr.re_evaluate

    # Pre-compute this

    aranges = np.arange(N+1, 0, -1)

    for R in range(N):
        br = block_length[R + 1]
        T_k = block_length[:R + 1] - br  # this looks like it is not used, but it actually is,
                                         # inside the numexpr expression

        # N_k: number of elements in each block
        # This expression has been simplified for the case of
        # unbinned events (i.e., one element in each block)
        # It was:
        #N_k = cumsum(x[:R + 1][::-1])[::-1]
        # Now it is:
        N_k = aranges[N - R:]
        # where aranges has been pre-computed

        # Evaluate fitness function
        # This is the slowest part, which I'm speeding up by using
        # numexpr. It provides a ~40% gain in execution speed.

        # The first time we need to "compile" the expression in numexpr,
        # all the other times we can reuse it

        if R == 0:

            fit_vec = numexpr_evaluate('''N_k * log(N_k/ T_k) ''',
                                       optimization='aggressive', local_dict={'N_k': N_k, 'T_k': T_k})

        else:

            fit_vec = numexpr_re_evaluate(local_dict={'N_k': N_k, 'T_k': T_k})

        A_R = fit_vec - prior  # type: np.ndarray

        A_R[1:] += best[:R]

        i_max = A_R.argmax()

        last[R] = i_max
        best[R] = A_R[i_max]

    numexpr.set_vml_accuracy_mode(oldaccuracy)

    logger.debug("Done\n")

    # Now peel off and find the blocks (see the algorithm in Scargle et al.)
    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N

    while True:

        i_cp -= 1

        change_points[i_cp] = ind

        if ind == 0:

            break

        ind = last[ind - 1]

    change_points = change_points[i_cp:]

    edg = edges[change_points]

    # Transform the found edges back into the original time system

    if (bkg_integral_distribution is not None):

        final_edges = map(lambda x: lookup_table[x], edg)

    else:

        final_edges = edg

    # Now fix the first and last edge so that they are tstart and tstop
    final_edges[0] = ttstart
    final_edges[-1] = ttstop

    return np.asarray(final_edges)


# To be run with a profiler
if __name__ == "__main__":

    tt = np.random.uniform(0, 1000, int(sys.argv[1]))
    tt.sort()

    with open("sim.txt", "w+") as f:
        for t in tt:
            f.write("%s\n" % (t))

    res = bayesian_blocks(tt, 0, 1000, 1e-3, None)
    print res
