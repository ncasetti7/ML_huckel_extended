# type: ignore
import sys
import traceback

class Suppressor(object):
    """
    Suppresses output to stdout
    """

    def __init__(self, suppress=True):
        self.suppress = suppress

    def __enter__(self):
        if self.suppress:
            self.stdout = sys.stdout
            sys.stdout = self

    def __exit__(self, _type, value, traceback):
        if self.suppress:
            sys.stdout = self.stdout
            if _type is not None:
                pass
            # Do normal exception handling

    def flush(self):
        pass

    def write(self, x):
        pass
