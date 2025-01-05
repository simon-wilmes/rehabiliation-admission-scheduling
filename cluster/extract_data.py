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
from pathlib import Path
from collections import defaultdict
from pprint import pprint
import json

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

output_dir = Path(os.path.abspath(os.path.dirname(__file__))).parent / "output"


def test_timeout(err_file_str):
    timeout_regex = "DUE TO TIME LIMIT"
    timeout_match = re.search(timeout_regex, err_file_str)
    if timeout_match:
        return True
    return False


def test_memory_limit(err_file_str):
    memory_regex = "Some of your processes may have been killed by the cgroup out-of-memory handler"
    memory_match = re.search(memory_regex, err_file_str)
    if memory_match:
        return True
    return False


def rec_dd():
    return defaultdict(rec_dd)


def extract_LBBD_solver(out_file, err_file):
    return {}


all_solv_par_inst_rep = rec_dd()
all_inst_solv_par_rep = rec_dd()


def extract_successful_data(out_file_str):
    time_in_solver = r"Time to find solution: (\d+(\.\d+)?)s with value (\d+(\.\d+)?)"
    total_time_regex = r"Total Time: (\d+(\.\d+)?)s"
    time_build_model_regex = r"Time to create model: (\d+(\.\d+)?)"
    time_build_model_match = re.search(time_build_model_regex, out_file_str).group(1)  # type: ignore
    time_in_solver_match = re.search(time_in_solver, out_file_str).group(1)  # type: ignore
    total_time_match = re.search(total_time_regex, out_file_str).group(1)  # type: ignore
    objective_value_match = re.search(time_in_solver, out_file_str).group(3)  # type: ignore
    return {
        "time_build_model": float(time_build_model_match),
        "time_in_solver": float(time_in_solver_match),
        "total_time": float(total_time_match),
        "objective_value": float(objective_value_match),
    }


def process_file(out_file, err_file):
    out_file_str = "\\n".join(out_file)
    err_file_str = "\\n".join(err_file)

    # Extract name params and instance

    name_regex = r"1.NAME: (.*)\n"
    params_regex = r"2.PARAMS: (.*)\n"
    instance_regex = r"3.INSTANCE: (.*)\n"
    rep_regex = r"4.REPETITION: (.*)\n"

    name_match = re.search(name_regex, out_file_str).group(1)  # type: ignore
    params_match = re.search(params_regex, out_file_str).group(1)  # type: ignore
    instance_match = re.search(instance_regex, out_file_str).group(1)  # type: ignore
    rep_match = re.search(rep_regex, out_file_str).group(1)  # type: ignore

    print(f"Name: {name_match}")
    print(f"Params: {params_match}")
    print(f"Instance: {instance_match}")
    print(f"Repetition: {rep_match}")
    data = {}

    if test_memory_limit(err_file_str):
        data = {"memory_limit": True}
    elif test_timeout(err_file_str):
        data = {"timeout": True}
    elif "All solvers accepted the solution." in out_file_str:
        successful_data = extract_successful_data(out_file_str)
        data.update(successful_data)

        match name_match:
            case "LBBDSolver":
                data.update(extract_LBBD_solver(out_file_str, err_file_str))
            case "MIPSolver":
                pass
            case "CPSolver":
                pass
            case "MIPSolver3":
                pass
            case _:
                raise ValueError(f"Solver {name_match} not recognized")

    all_solv_par_inst_rep[name_match][params_match][instance_match][rep_match] = data
    all_inst_solv_par_rep[instance_match][name_match][params_match][rep_match] = data


# Loop through all files in the directory
for file in os.listdir(output_dir):
    if file.endswith(".out"):
        file_path = output_dir / file
        # Process the file
        print(f"Processing file: {file_path}")
        with open(file_path, "r") as f:
            out_file = f.readlines()
        try:
            err_file_path = str(file_path).replace(".out", ".err")
            with open(err_file_path, "r") as f:
                err_file = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Error file not found: {err_file_path}")

        process_file(out_file, err_file)


def recursive_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):  # Check if the current level is a defaultdict
        return {key: recursive_defaultdict_to_dict(value) for key, value in d.items()}
    return d  # Base case: return the value as is if it's not a defaultdict


all_solv_par_inst_rep = recursive_defaultdict_to_dict(all_solv_par_inst_rep)
all_inst_solv_par_rep = recursive_defaultdict_to_dict(all_inst_solv_par_rep)


def pprint_dict(d):

    print(json.dumps(d, indent=4, sort_keys=True))


all_solv_par_inst_rep.pop("LBBDSolver")
all_solv_par_inst_rep.pop("CPSolver")


def check_if_solved(d):
    if isinstance(d, dict):
        if "objective_value" in d:
            return 1

        sum_v = 0
        for key in d:
            sum_v += check_if_solved(d[key])
        return sum_v
    return 0


instance_solution_count = defaultdict(int)
for instance, solvers in all_inst_solv_par_rep.items():
    if val := check_if_solved(solvers):
        print(f"Instance {instance} has at least {val} solver that found a solution.")
        instance_str = "_".join(instance.split("_")[:-1])
        instance_solution_count[instance_str] += 0
    else:
        print(f"Instance {instance} has no solver that found a solution.")
        instance_str = "_".join(instance.split("_")[:-1])
        instance_solution_count[instance_str] += 1

pprint_dict(instance_solution_count)
# MIPSolver3 << MIPSolver


pprint_dict(all_solv_par_inst_rep)
input("Continue")
pass
pprint_dict(all_inst_solv_par_rep)
