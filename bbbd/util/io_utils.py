import os


def sanitize_filename(filename):

    return os.path.abspath(os.path.expandvars(os.path.expanduser(filename)))