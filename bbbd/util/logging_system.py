import logging
import time
import os
import re

_format = '%(name)15s: %(message)s'


class CustomHandler(logging.StreamHandler):
    """
    A custom log handler that splits messages based on the number of charachters

    """

    def __init__(self, limit=60):

        self._limit = int(limit)

        self._regexp = re.compile('.{1,60}')

        super(CustomHandler, self).__init__()

    def format(self, record):

        lines = re.findall(self._regexp, record.msg)

        formatted = []

        for message in lines:

            record.msg = message

            this_formatted = super(CustomHandler, self).format(record)

            formatted.append(this_formatted)

        return "\n".join(formatted)


def get_logger(name):

    # Setup the logger

    logger = logging.getLogger(os.path.splitext(name)[0])

    logger.setLevel(logging.INFO)

    # Prepare the handler

    handler = CustomHandler(65)

    formatter = logging.Formatter(_format)

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logger.propagate = False

    logger.info("Setup logger %s at %s" % (name, time.asctime()))

    return logger
