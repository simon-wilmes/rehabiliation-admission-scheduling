from typing import TypeVar, Dict, Any
import os
from datetime import datetime
from itertools import product, combinations
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
from collections import defaultdict


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


def calculate_similarity_scores(sim_list: list[list[dict]]):
    out_list1 = defaultdict(int)
    out_list2 = defaultdict(int)
    for list1, list2 in combinations(sim_list, r=2):
        changes, num_changes = calculate_dict_changes(list1, list2)
        out_list1[changes] += 1
        out_list2[num_changes] += 1
    return out_list1, out_list2


def calculate_dict_changes(set1, set2):
    """
    Calculate minimum changes needed to make two sets of dictionaries identical.

    Args:
        set1: List[Dict[int, int]] - First set of dictionaries
        set2: List[Dict[int, int]] - Second set of dictionaries

    Returns:
        tuple(int, int) - (total changes needed, number of dictionaries that need changes)
    """
    # Convert dictionaries to frozenset of items for hashability
    frozen_set1 = {frozenset(d.items()) for d in set1}
    frozen_set2 = {frozenset(d.items()) for d in set2}

    # Find common dictionaries
    common = frozen_set1 & frozen_set2

    # Remove common dictionaries
    remaining1 = frozen_set1 - common
    remaining2 = frozen_set2 - common

    # Convert back to list of dicts for easier processing
    rem1 = [dict(d) for d in remaining1]
    rem2 = [dict(d) for d in remaining2]

    # If sets are unequal size, pad smaller one with empty dicts
    while len(rem1) < len(rem2):
        rem1.append({})
    while len(rem2) < len(rem1):
        rem2.append({})

    if not rem1 and not rem2:
        return 0, 0

    # Calculate distances between all remaining dictionaries
    distances = []
    for i, dict1 in enumerate(rem1):
        for j, dict2 in enumerate(rem2):
            # Calculate changes needed between these dictionaries
            changes = calculate_dict_distance(dict1, dict2)
            distances.append((changes, i, j))

    # Sort by number of changes needed
    distances.sort()

    # Match dictionaries greedily
    used1 = set()
    used2 = set()
    total_changes = 0
    pairs = []

    for changes, i, j in distances:
        if i not in used1 and j not in used2:
            used1.add(i)
            used2.add(j)
            total_changes += changes
            pairs.append((i, j))

            if len(used1) == len(rem1):
                break

    return total_changes, len(pairs)


def calculate_dict_distance(dict1, dict2):
    """
    Calculate minimum changes needed to transform dict1 into dict2.

    Args:
        dict1: Dict[int, int] - First dictionary
        dict2: Dict[int, int] - Second dictionary

    Returns:
        int - Number of changes needed
    """
    changes = 0

    # Count keys that need to be added or changed
    for key, value in dict2.items():
        if key not in dict1 or dict1[key] != value:
            changes += 1

    # Count keys that need to be removed
    for key in dict1:
        if key not in dict2:
            changes += 1

    return changes
