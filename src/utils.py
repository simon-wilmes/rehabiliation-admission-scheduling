from typing import TypeVar, Dict, Any
import os
from datetime import datetime
from itertools import product
import sys
import io

CP_PATH = (
    "/home/simon/ibm/ILOG/CPLEX_Studio2211/cpoptimizer/bin/x86-64_linux/cpoptimizer"
)


A = TypeVar("A")
B = TypeVar("B")


def slice_dict(d: Dict[A, B], conditions: tuple) -> dict:
    return {
        k: v
        for k, v in d.items()
        if all(
            cond is None or k[i] == cond for i, cond in enumerate(conditions)  # type: ignore
        )
    }


class NoOpWriter:
    def write(self, data):
        pass

    def flush(self):
        pass


import os
import sys
import threading
import time


class OutputGrabber(object):
    """
    Class used to grab standard output or another stream.
    """

    escape_char = "\b"

    def __init__(self, stream=None, threaded=False):
        self.origstream = stream
        self.threaded = threaded
        if self.origstream is None:
            self.origstream = sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """
        Start capturing the stream data.
        """
        self.capturedtext = ""
        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)
        if self.threaded:
            # Start thread that will read the stream:
            self.workerThread = threading.Thread(target=self.readOutput)
            self.workerThread.start()
            # Make sure that the thread is running and os.read() has executed:
            time.sleep(0.01)

    def stop(self):
        """
        Stop capturing the stream data and save the text in `capturedtext`.
        """
        # Print the escape character to make the readOutput method stop:
        self.origstream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.origstream.flush()
        if self.threaded:
            # wait until the thread finishes so we are sure that
            # we have until the last character:
            self.workerThread.join()
        else:
            self.readOutput()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `capturedtext`.
        """
        while True:
            char = os.read(self.pipe_out, 1).decode(self.origstream.encoding)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char


def get_file_writer_context(log_to_console: bool, **kwargs):
    # We no longer have multiple streams. Just return a buffered stdout if logging to console.
    if log_to_console:
        return OutputGrabber()
    else:
        return NoOpWriter()


def generate_combis(solver, allowed_keys):
    d = solver.SOLVER_OPTIONS
    restr_d = {key: value for key, value in d.items() if key in allowed_keys}
    assert set(allowed_keys) <= set(d.keys()), "Not all keys are in the solver options"

    keys = list(restr_d.keys())
    value_lists = [restr_d[k] for k in keys]
    # Use itertools.product to create all combinations
    combinations = []
    for combo in product(*value_lists):
        # zip the keys and chosen values to form a new dictionary
        combination_dict = dict(zip(keys, combo))
        combinations.append(combination_dict)
    return combinations
