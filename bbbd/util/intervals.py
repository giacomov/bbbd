
class IntervalsDoNotOverlap(RuntimeError):
    pass


class IntervalsNotContiguous(RuntimeError):
    pass


class Interval(object):
    def __init__(self, start, stop, swap_if_inverted=False):

        self._start = float(start)
        self._stop = float(stop)

        # Note that this allows to have intervals of zero duration

        if self._stop < self._start:

            if swap_if_inverted:

                self._start = stop
                self._stop = start

            else:

                raise RuntimeError("Invalid time interval! TSTART must be before TSTOP and TSTOP-TSTART >0. "
                                   "Got tstart = %s and tstop = %s" % (start, stop))

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @classmethod
    def new(cls, *args, **kwargs):

        return cls(*args, **kwargs)

    def _get_width(self):

        return self._stop - self._start

    @property
    def mid_point(self):

        return (self._start + self._stop) / 2.0

    def __repr__(self):

        return " interval %s - %s (width: %s)" % (self.start, self.stop, self._get_width())

    def intersect(self, interval):
        """
        Returns a new time interval corresponding to the intersection between this interval and the provided one.

        :param interval: a TimeInterval instance
        :type interval: Interval
        :return: new interval covering the intersection
        :raise IntervalsDoNotOverlap : if the intervals do not overlap
        """

        if not self.overlaps_with(interval):
            raise IntervalsDoNotOverlap("Current interval does not overlap with provided interval")

        new_start = max(self._start, interval.start)
        new_stop = min(self._stop, interval.stop)

        return self.new(new_start, new_stop)

    def merge(self, interval):
        """
        Returns a new interval corresponding to the merge of the current and the provided time interval. The intervals
        must overlap.

        :param interval: a TimeInterval instance
         :type interval : Interval
        :return: a new TimeInterval instance
        """

        if self.overlaps_with(interval):

            new_start = min(self._start, interval.start)
            new_stop = max(self._stop, interval.stop)

            return self.new(new_start, new_stop)

        else:

            raise IntervalsDoNotOverlap("Could not merge non-overlapping intervals!")

    def overlaps_with(self, interval):
        """
        Returns whether the current time interval and the provided one overlap or not

        :param interval: a TimeInterval instance
        :type interval: Interval
        :return: True or False
        """

        if interval.start == self._start or interval.stop == self._stop:

            return True

        elif interval.start > self._start and interval.start < self._stop:

            return True

        elif interval.stop > self._start and interval.stop < self._stop:

            return True

        elif interval.start < self._start and interval.stop > self._stop:

            return True

        else:

            return False

    def to_string(self):
        """
        returns a string representation of the time interval that is like the
        argument of many interval reading funcitons

        :return:
        """

        return "%f-%f" % (self.start, self.stop)

    def __eq__(self, other):

        if not isinstance(other, Interval):

            # This is needed for things like comparisons to None or other objects.
            # Of course if the other object is not even a TimeInterval, the two things
            # cannot be equal

            return False

        else:

            return self.start == other.start and self.stop == other.stop


class TimeInterval(Interval):

    def __add__(self, number):
        """
        Return a new time interval equal to the original time interval shifted to the right by number

        :param number: a float
        :return: a new TimeInterval instance
        """

        return self.new(self._start + number, self._stop + number)

    def __sub__(self, number):
        """
        Return a new time interval equal to the original time interval shifted to the left by number

        :param number: a float
        :return: a new TimeInterval instance
        """

        return self.new(self._start - number, self._stop - number)

    @property
    def duration(self):

        return super(TimeInterval, self)._get_width()

    @property
    def start_time(self):

        return self._start

    @property
    def stop_time(self):

        return self._stop

    @property
    def half_time(self):

        return self.mid_point

    def __repr__(self):

        return "time interval %s - %s (duration: %s)" % (self.start_time, self.stop_time, self.duration)
