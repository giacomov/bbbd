import json

from util.io_utils import sanitize_filename


class ResultsContainer(object):
    """
    A class to contain the results of the search

    """
    _default_values = {'name': '',
                       'trigger time': 0.0,
                       'ra': 999,
                       'dec': 999,
                       'background fit gof': -1,
                       'number of intervals': 0,
                       'detected': False,
                       'blocks': '',
                       'highest net rate': -1.0,
                       'highest net rate error': -1.0,
                       'highest net rate tstart': 999,
                       'highest net rate tstop': 999,
                       'highest net rate duration': -1,
                       'highest net rate background': -1.0
                       }

    def __init__(self):

        self._inner_dict = dict(ResultsContainer._default_values)

    def __setitem__(self, key, value):

        assert key in self._inner_dict, "Key %s is not a valid key for results" % key

        self._inner_dict[key] = value

    def __getitem__(self, item):

        return self._inner_dict[item]

    def write_to(self, json_file):
        """
        Write results to a JSON file (overwriting it if it exists)

        :param json_file:
        :return:
        """

        with open(sanitize_filename(json_file), "w+") as f:

            # Write into file with pretty printing

            json.dump(self._inner_dict, f, sort_keys=True, indent=4, separators=(',', ': '))
