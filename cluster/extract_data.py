try:
    import pyperclip

    pyperclip_available = True
except ImportError:
    pyperclip_available = False


# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import re

# import numpy as np


def set_clipboard(text):
    if pyperclip_available:
        pyperclip.copy(text)


matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "axes.labelsize": "x-large",
    }
)


directory = "../output"


def process_file(out_file, err_file):
    out_file_str = "\\n".join(out_file)
    err_file_str = "\\n".join(err_file)

    # Extract name params and instance

    name_regex = "1.NAME: (.*)\\n"
    params_regex = "2.PARAMS: (.*)\\n"
    instance_regex = "3.INSTANCE: (.*)\\n"
    rep_regex = "4.REPETITION: (.*)\\n"

    name_match = re.search(name_regex, out_file_str).group(1)  # type: ignore
    params_match = re.search(params_regex, out_file_str).group(1)  # type: ignore
    instance_match = re.search(instance_regex, out_file_str).group(1)  # type: ignore
    rep_match = re.search(rep_regex, out_file_str).group(1)  # type: ignore
    pass


# Loop through all files in the directory
for file in os.listdir(directory):
    if file.endswith(".out"):
        file_path = os.path.join(directory, file)
        # Process the file
        print(f"Processing file: {file_path}")
        with open(file_path, "r") as f:
            out_file = f.readlines()
        try:
            err_file_path = file_path.replace(".out", ".err")
            with open(err_file_path, "r") as f:
                err_file = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Error file not found: {err_file_path}")

        process_file(out_file, err_file)
