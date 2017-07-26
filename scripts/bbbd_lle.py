#!/usr/bin/env python

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")

import argparse
import os
import sys
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt

import GtApp

from bbbd.event_histogram import EventHistogram, NoDataToFit
from bbbd.util.logging_system import get_logger
from bbbd.util.io_utils import sanitize_filename
from bbbd.fit_functions.lle_polynomial import LLEPolynomial
from bbbd.statistic.bayesian_blocks import bayesian_blocks
from bbbd.statistic.bkg_integral_distribution import BackgroundIntegralDistribution
from bbbd.results_container import ResultsContainer
from bbbd.statistic.significance import Significance
from bbbd.instrument_specific.Fermi_LAT_LLE import NoGTI

def get_rate(arrival_times, exposure_function, bins, reference_time, bkg_poly=None, best_fit_poly=None):

    # Now compute the rates of events
    counts, _ = np.histogram(arrival_times, bins=bins)

    # Transform bins in MET

    blocks_start = bins[:-1]
    blocks_stop = bins[1:]

    # Get the exposure for each block

    exposure = np.array(map(lambda (t1, t2): exposure_function(t1 + reference_time, t2 + reference_time),
                            zip(blocks_start, blocks_stop)))

    # Get the rates
    idx = exposure > 0

    rates = np.zeros_like(counts, dtype=float)

    rates[idx] = counts[idx] / exposure[idx]

    rates_errors = np.zeros_like(rates)
    rates_errors[idx] = np.sqrt(counts[idx]) / exposure[idx]

    if bkg_poly is None:

        return rates, rates_errors

    else:

        # Subtrack the background
        bkg_rate = bkg_poly(blocks_start[idx], blocks_stop[idx], best_fit_poly)
        bkg_sub_rate = np.zeros_like(rates)
        bkg_sub_rate[idx] = rates[idx] - bkg_rate

        return bkg_sub_rate, rates_errors


def go(args):

    # Get the logger
    logger = get_logger(os.path.basename(__file__))

    # Check input parameters and sanitize file names

    lle_file_orig = sanitize_filename(args.lle)
    ft2_file = sanitize_filename(args.pt)

    assert os.path.exists(lle_file_orig), "Provided LLE (FT1) file %s does not exist" % lle_file_orig
    assert os.path.exists(ft2_file), "Provided pointing file (FT2) %s does not exist" % ft2_file

    # The interval edges should be even
    assert len(args.off_pulse_intervals) % 2 == 0, "You have to provide an even number of edges for the off pulse " \
                                                   "interval"

    assert len(args.search_window) == 2, "The search window should be a list of two floats (start and end of window)"

    # Get the coordinates of the object
    ra = pyfits.getval(lle_file_orig, "RA_OBJ")
    dec = pyfits.getval(lle_file_orig, "DEC_OBJ")

    # First apply a GTI cut on the LLE file, since the GTI inside the FT1 are not good
    # (for example, they do not exclude intervals with livetime=0)

    # But first correct the GTIs (exclude intervals where livetime = 0)
    gtmktime = GtApp.GtApp("gtmktime")

    # Unique file name

    root = os.path.splitext(os.path.basename(lle_file_orig))[0]
    lle_file = "%s_mkt.fit" % root

    logger.info("Running gtmktime")

    try:
        gtmktime.run(scfile=ft2_file,
                     filter="(DATA_QUAL>0 || DATA_QUAL==-1) && LAT_CONFIG==1 && IN_SAA!=T && LIVETIME>0 && "
                            "ANGSEP(RA_SCZ, DEC_SCZ, %.3f, %.3f) < %s && "
                            "ANGSEP(RA_ZENITH, DEC_ZENITH, %.3f, %.3f) < 110" % (ra, dec, args.theta_max, ra, dec),
                     roicut="no",
                     evfile=lle_file_orig,
                     outfile=lle_file,
                     apply_filter='yes',
                     overwrite='yes')

    except:

        raise RuntimeError("gtmktime failed!")

    else:

        logger.info("gtmktime done")

    # Create container for the results
    results = ResultsContainer()

    # Read information on the event from the FITS file

    ra = pyfits.getval(lle_file, "RA_OBJ")
    dec = pyfits.getval(lle_file, "DEC_OBJ")
    trigger_time = pyfits.getval(lle_file, "TRIGTIME")
    trigger_name = pyfits.getval(lle_file, "OBJECT")

    results['name'] = trigger_name
    results['trigger time'] = trigger_time
    results['ra'] = ra
    results['dec'] = dec

    # Print out the information we just gathered


    logger.info("Read trigger information from %s" % lle_file)
    logger.info("Trigger name: %s" % trigger_name)
    logger.info("(RA, Dec) = (%.3f, %.3f)" % (ra, dec))
    logger.info("Trigger time = %.3f (MET)" % trigger_time)

    # Make histogram and fit the background
    try:

        eh = EventHistogram.from_ft1_and_ft2(lle_file, ft2_file, bin_size=args.binsize,
                                             selection=args.cut)

    except NoGTI:

        logger.error("There are no GTIs in ft1 file %s" % lle_file)

        _clean_exit(results, logger, args.outfile, status="No GTIs in FT1 file")

    # Use the polynomial multiplied by the cos(theta) (theta is the off-axis angle)

    llep = LLEPolynomial(ra=ra, dec=dec, trigger_time=trigger_time, ft2file=ft2_file, degree=args.poly_degree)

    # Format the off_pulse_intervals from [-400, -20, 150, 500] to ((-400, -20), (150, 500))

    raw_off_pulse_intervals = zip(args.off_pulse_intervals[::2], args.off_pulse_intervals[1::2])

    # Make sure that the off pulse intervals have at least a bit of exposure
    off_pulse_intervals = []

    for (t1, t2) in raw_off_pulse_intervals:

        in_gti, new_t1_met, new_t2_met = eh.lle_exposure.is_interval_in_gti(t1 + trigger_time, t2 + trigger_time)

        this_expo = eh.lle_exposure.get_exposure(new_t1_met, new_t2_met)

        # Accept the background interval if it has at least 5% of the exposure of the proposed interval
        # This is to avoid keeping intervals that are too short

        if in_gti and this_expo >= (t2 - t1) / 100.0 * 5:

            logger.info("Off-pulse interval %.3f - %.3f accepted as %.3f - %.3f" % (t1,
                                                                                    t2,
                                                                                    new_t1_met - trigger_time,
                                                                                    new_t2_met - trigger_time))

            off_pulse_intervals.append((new_t1_met - trigger_time, new_t2_met - trigger_time))

        else:

            logger.error("Off-pulse interval %.3f - %.3f has been rejected. In GTI: %s, exposure: %.3f" % (t1,
                                                                                                           t2,
                                                                                                           in_gti,
                                                                                                           this_expo))

            logger.error("One or more of the off-pulse intervals are not in GTIs or have an exposure too small. "
                         "Cannot continue.")

            _clean_exit(results, logger, args.outfile, status="One or more off-pulse intervals have been rejected")

    # Fit the background

    try:

        best_fit_poly, cstat, fig = eh.fit_background(llep, off_pulse_intervals)

    except NoDataToFit:

        logger.error("The background selection returned zero bins. Cannot fit the background. Exiting.")

        _clean_exit(results, logger, args.outfile, status="No bins survive the exposure and GTI cut for the background")

    # Compute goodness of fit for the background
    # (this also returns the best fit parameters for the polynomials obtained from the simulations)
    bkg_gof, bkg_sim_best_fits = eh.get_background_gof(cstat, llep, best_fit_poly, args.nbkgsim,
                                                       off_pulse_intervals)

    logger.info("Background goodness of fit: %.3f" % (bkg_gof))

    # Save into results

    results['background fit gof'] = bkg_gof

    # Save background fit plot

    bkg_plot_file = sanitize_filename("bkgfit_%s.png" % trigger_name)

    logger.info("Saving background fit plot to %s" % bkg_plot_file)

    fig.savefig(bkg_plot_file)

    # Setup search window

    search_tstart, search_tstop = args.search_window

    # Check whether the search window is within a GTI,
    # If it is but only partially, the search window will be adjusted

    within_gti, search_tstart, search_tstop = eh.lle_exposure.is_interval_in_gti(search_tstart + trigger_time,
                                                                                 search_tstop + trigger_time)

    # Get the exposure of the window
    window_exposure = eh.lle_exposure.get_exposure(search_tstart, search_tstop)

    # The updated values are in MET, need to remove the trigger time

    search_tstart -= trigger_time
    search_tstop -= trigger_time

    if not within_gti or window_exposure <= 0:

        logger.error("The search window is outside the GTIs. Cannot continue.")

        _clean_exit(results, logger, args.outfile, status="Search window is completely outside the GTIs")

    # First we need the integral distribution of the background
    bkg_int_distr = BackgroundIntegralDistribution(llep, best_fit_parameters=best_fit_poly,
                                                   tstart=search_tstart, tstop=search_tstop)

    # Create a mask to select only the events within the search window
    idx = (eh.arrival_times >= search_tstart) & (eh.arrival_times < search_tstop)

    selected_events = eh.arrival_times[idx]

    selected_events.sort()

    logger.info("Selected %i events within the time window %.3f - %.3f" % (selected_events.shape[0],
                                                                           search_tstart, search_tstop))

    if selected_events.shape[0] < 2:

        logger.error("Too few events selected. Nothing to do.")

        _clean_exit(results, logger, args.outfile, status="Too few events in search window")

    # Run the Bayesian Blocks

    logger.info("Running Bayesian Blocks...")

    blocks = bayesian_blocks(selected_events, search_tstart, search_tstop, args.p0, bkg_int_distr)

    n_intervals = len(blocks) - 1

    bb_file = sanitize_filename("bb_res_%s.png" % trigger_name)

    detected = n_intervals > 2

    # Save into results

    results['number of intervals'] = n_intervals
    results['detected'] = detected
    results['blocks'] = ",".join(map(lambda x: "%.3f" % x, blocks))

    if detected:

        interesting_intervals = zip(blocks[1:-1], blocks[2:-1])

        # Print out the off_pulse_intervals (if any)
        logger.info("Found %i interesting intervals between %.3f and %.3f" % (len(interesting_intervals),
                                                                              search_tstart, search_tstop))

        # Make image and print off_pulse_intervals
        fig = eh.plot()

        for t1, t2 in interesting_intervals:

            logger.info("%.3f - %.3f" % (t1, t2))

            for tt in [t1, t2]:

                fig.axes[0].axvline(tt, linestyle='--')

        # Cover only the search window

        fig.axes[0].set_xlim([search_tstart, search_tstop])

        logger.info("Saving light curve to %s" % bb_file)

        fig.savefig(bb_file)

        # Now compute the rates of events
        bkg_sub_rate, bkg_sub_rate_err = get_rate(selected_events, eh.lle_exposure.get_exposure, blocks,
                                                  trigger_time, bkg_poly=llep, best_fit_poly=best_fit_poly)

        # Compute the observed counts in the blocks
        obs_counts_in_blocks, _ = np.histogram(selected_events, blocks)

        # Find the interval with the highest rate
        max_rate_idx = bkg_sub_rate.argmax()

        # Get its start and stop time
        max_rate_tstart = blocks[:-1][max_rate_idx]
        max_rate_tstop = blocks[1:][max_rate_idx]
        max_rate_duration = max_rate_tstop - max_rate_tstart

        # Get maximum rate and rate error
        highest_net_rate = bkg_sub_rate[max_rate_idx]
        highest_net_rate_err = bkg_sub_rate_err[max_rate_idx]

        # Get corresponding background and background error
        highest_net_rate_bkg = llep(max_rate_tstart, max_rate_tstop, best_fit_poly)

        # To get the error we use the best fit values of the simulations obtained above and measure
        # the standard deviation
        bkg_estimates = map(lambda this_best_fit:llep(max_rate_tstart, max_rate_tstop, this_best_fit),
                            bkg_sim_best_fits)

        highest_net_rate_bkg_err = np.std(bkg_estimates)

        # Now compute the significance
        this_expo = eh.lle_exposure.get_exposure(max_rate_tstart + trigger_time,
                                                 max_rate_tstop + trigger_time)
        sig = Significance(obs_counts_in_blocks[max_rate_idx],
                           highest_net_rate_bkg * this_expo, alpha=1.0)

        highest_net_rate_significance = sig.li_and_ma_equivalent_for_gaussian_background(highest_net_rate_bkg_err *
                                                                                         this_expo)[0]

        logger.info("Maximum net rate: %.3f +/- %.3f cts at %.3f - %.3f "
                    "(duration = %.2f s)" % (highest_net_rate,
                                             highest_net_rate_err,
                                             max_rate_tstart, max_rate_tstop,
                                             max_rate_duration))
        logger.info("Significance at maximum rate: %.2f sigma" % highest_net_rate_significance)
        logger.info("Raw counts in maximum rate interval: %i" % obs_counts_in_blocks[max_rate_idx])

        # Save in the results
        results['highest net rate'] = highest_net_rate
        results['highest net rate error'] = highest_net_rate_err
        results['highest net rate tstart'] = max_rate_tstart
        results['highest net rate tstop'] = max_rate_tstop
        results['highest net rate duration'] = max_rate_duration
        results['highest net rate exposure'] = this_expo
        results['highest net rate background'] = highest_net_rate_bkg
        results['highest net rate background error'] = highest_net_rate_bkg_err
        results['highest net rate significance'] = highest_net_rate_significance

        # Make a light curve with a bin size equal to the length of the block with the maximum rate,
        # and shifted so that the block with the maximum rate is exactly one of the bins
        optimal_bins = np.arange(max_rate_tstart - 10 * max_rate_duration,
                                 max_rate_tstop + 10 * max_rate_duration,
                                 max_rate_duration)

        bkg_sub_rate, bkg_sub_rate_err = get_rate(selected_events, eh.lle_exposure.get_exposure, optimal_bins,
                                                  trigger_time, bkg_poly=llep, best_fit_poly=best_fit_poly)

        # Plot the optimal light curve
        fig, sub = plt.subplots(1, 1)

        rr = np.zeros_like(optimal_bins, dtype=float)
        rr[:-1] = bkg_sub_rate

        sub.step(optimal_bins, rr, where='post', color='blue')

        bc = 0.5 * (optimal_bins[1:] + optimal_bins[:-1])
        sub.errorbar(bc, rr[:-1], yerr=bkg_sub_rate_err, fmt='.', color='blue')

        sub.axhline(0, linestyle='--', color='grey', alpha=0.5)

        sub.set_ylabel("Net rate (cts/s)")
        sub.set_xlabel("Time since %s" % trigger_time)

        # Zoom in if necessary so we do not show parts where the background model does not apply
        # (i.e., before and after the end of the off-pulse intervals)
        sub.set_xlim([max(search_tstart, min(args.off_pulse_intervals), optimal_bins.min()),
                      min(search_tstop, max(args.off_pulse_intervals), optimal_bins.max())])

        optimal_lc = sanitize_filename("optimal_lc_%s.png" % trigger_name)

        logger.info("Saving optimal light curve %s" % optimal_lc)

        fig.savefig(optimal_lc)

    else:

        # No detection

        logger.info("No interesting interval found.")

        # Now compute the rates of events

        optimal_bins = np.arange(min(args.off_pulse_intervals),
                                 max(args.off_pulse_intervals),
                                 1.0)

        bkg_sub_rate, bkg_sub_rate_err = get_rate(eh.arrival_times, eh.lle_exposure.get_exposure, optimal_bins,
                                                  trigger_time, bkg_poly=llep, best_fit_poly=best_fit_poly)

        # Plot the optimal light curve
        fig, sub = plt.subplots(1, 1)

        rr = np.zeros_like(optimal_bins, dtype=float)
        rr[:-1] = bkg_sub_rate

        sub.step(optimal_bins, rr, where='post', color='blue')

        sub.axhline(0, linestyle='--', color='grey', alpha=0.5)

        sub.set_ylabel("Net rate (cts/s)")
        sub.set_xlabel("Time since %s" % trigger_time)

        # Zoom in if necessary so we do not show parts where the background model does not apply
        # (i.e., before and after the end of the off-pulse intervals)
        sub.set_xlim([max(optimal_bins.min(), min(args.off_pulse_intervals)),
                      min(optimal_bins.max(), max(args.off_pulse_intervals))])

        optimal_lc = sanitize_filename("optimal_lc_%s.png" % trigger_name)

        logger.info("Saving optimal light curve %s" % optimal_lc)

        fig.savefig(optimal_lc)

    _clean_exit(results, logger, args.outfile, status="success")


def _clean_exit(results, logger, outfile, status):

    # Update the status before writing it to file

    results['final status'] = status

    # Write JSON file
    outfile = sanitize_filename(outfile)
    logger.info("Writing results in JSON to %s" % (outfile))
    results.write_to(outfile)

    # Print the results
    print("\nFinal results:")
    print("==============\n")
    results.display()
    print("\n")

    sys.exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Find transients in LLE data using Bayesian Blocks')

    parser.add_argument("--lle", help="Path to the LLE (FT1) file", type=str, required=True)
    parser.add_argument("--pt", help="Path to the pointing file (FT2)", type=str, required=True)
    parser.add_argument("--p0", help="False detection probability for the Bayesian Block step (default: 1e-4)",
                        type=float, default=1e-4)
    parser.add_argument("--outfile", help="Outfile which will contain the results of the analysis",
                        type=str, required=True)
    parser.add_argument("--search_window", help="Time interval where to search for a signal (search window). "
                                                "Default is '-20 200'",
                        nargs='+', default=[-20.0, 200.0], type=float)
    parser.add_argument("--binsize", help="Binsize for the histogram used to fit the background (default: 1.0)",
                        default=1.0, type=float)
    parser.add_argument("--cut", help="Cut for the data. It is a string like '(ENERGY > 10) & (ENERGY < 100)', where"
                                      "any name of a column in the LLE (FT1) file can be used. Default: None",
                        default=None, type=str)
    parser.add_argument("--theta_max", help="Maximum theta. Time intervals where the source is at an off-axis angle "
                                            "larger than this value will be removed from the analysis",
                                       default=88, type=float)
    parser.add_argument("--poly_degree", help="Degree for the polynomial to be used in the fit (default: 3)",
                        default=3, type=int)
    parser.add_argument("--nbkgsim", help="Number of simulations for the background goodness of fit (default: 1000)",
                        default=1000, type=int)
    parser.add_argument("--off_pulse_intervals", help="Definition of the off-pulse off_pulse_intervals for background fitting."
                                                      "Default is '-400 -20 150 500' corresponding to the off_pulse_intervals"
                                                      "-400 - 20 and 150 - 500 (in time since trigger).",
                        default=[-400.0, -20.0, 150.0, 500.0], type=float, nargs='+')

    args = parser.parse_args()

    go(args)