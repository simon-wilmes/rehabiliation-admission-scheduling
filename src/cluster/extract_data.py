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

replace_words = {
    r"\_less\_prescriptions": r"\_a",
    r"\_more\_resources": r"\_b",
    r"\_short\_timeframe": r"\_c",
}

remove_runs = {'{"break_symmetry": true}'}

remove_runs_contains = {
    '"substitute_x_pmdt": true',
    '"break_symmetry": true, "subsol',
    '"add_constraints_to_symmetric_days": false, "b',
    '"subsolver.restrict_obj_func_to_1": false',
    '"use_lazy_constraints": false',
}


output_dir = (
    Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
    / "output"
    / "study_2"
)


def remove_keys_with_name(data, key_to_remove, contains=False):
    if isinstance(data, dict):
        # Use dictionary comprehension to filter out the key
        return {
            key: remove_keys_with_name(value, key_to_remove, contains)
            for key, value in data.items()
            if (not contains and key != key_to_remove)
            or (contains and key_to_remove not in str(key))
        }
    elif isinstance(data, list):
        # Recursively process list elements
        return [remove_keys_with_name(item, key_to_remove, contains) for item in data]
    return data


# import numpy as np
def pprint_dict(d):

    print(json.dumps(d, indent=4, sort_keys=True))


def hash_dict(d):
    return json.dumps(d, sort_keys=True)


def set_clipboard(text):
    if pyperclip_available:
        pyperclip.copy(text)


def main():

    unwanted_keys = {}

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

    def extract_utilization(out_file_str):
        utilization_regex = r"Average utilization factor: (\d+(\.\d+)?)"
        utilization_match = re.findall(utilization_regex, out_file_str)

        all_util_regex = r"Resource (\d+): (\d+(\.\d+)?)"
        all_util_match = re.findall(all_util_regex, out_file_str)

        if utilization_match:
            # find min and max utilization and median
            util_list = [float(x[1]) for x in all_util_match]
            util_per_res = {int(x[0]): float(x[1]) for x in all_util_match}
            min_util = min(util_list)
            max_util = max(util_list)
            median_util = sorted(util_list)[len(util_list) // 2]
            return {
                "average_utilization": float(utilization_match[-1][0]),
                "min_util": min_util,
                "max_util": max_util,
                "median_util": median_util,
                # "util_per_res": util_per_res,
            }
        return {}

    def rec_dd():
        return defaultdict(rec_dd)

    def extract_LBBD_solver(out_file_str, err_file_str):
        create_model_regex = r"Subsolver: Create Model: (\d+(\.\d+)?)/"
        solve_model_regex = r"Subsolver: Solve Model: (\d+(\.\d+)?)/"

        create_model_match = re.findall(create_model_regex, out_file_str)[-1][0]  # type: ignore
        solve_model_match = re.findall(solve_model_regex, out_file_str)[-1][0]  # type: ignore

        return {
            "subsolver_create_model_time": float(create_model_match),
            "subsolver_solve_model_time": float(solve_model_match),
        }

    all_solv_par_inst_rep = rec_dd()
    all_inst_solv_par_rep = rec_dd()

    def extract_successful_data(out_file_str):
        time_in_solver = (
            r"Time to find solution: (\d+(\.\d+)?)s with value (\d+(\.\d+)?)"
        )
        total_time_regex = r"Total Time: (\d+(\.\d+)?)s"
        time_build_model_regex = r"Time to create model: (\d+(\.\d+)?)"
        time_build_model_match = re.findall(time_build_model_regex, out_file_str)[-1][0]  # type: ignore
        time_in_solver_match = re.findall(time_in_solver, out_file_str)[-1][0]  # type: ignore
        total_time_match = re.findall(total_time_regex, out_file_str)[-1][0]  # type: ignore
        objective_value_match = re.findall(time_in_solver, out_file_str)[-1][2]  # type: ignore
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
        job_name_regex = r"0.JOB_NAME: (.*)\n"
        name_regex = r"1.SOLVER: (.*)\n"
        params_regex = r"2.PARAMS: (.*)\n"
        instance_regex = r"3.INSTANCE: (.*)\n"
        rep_regex = r"4.REPETITION: (.*)\n"

        job_name_match = re.search(job_name_regex, out_file_str).group(1)  # type: ignore
        name_match = re.search(name_regex, out_file_str).group(1)  # type: ignore
        params_match = eval(re.search(params_regex, out_file_str).group(1))  # type: ignore
        instance_match = re.search(instance_regex, out_file_str).group(1)  # type: ignore
        rep_match = re.search(rep_regex, out_file_str).group(1)  # type: ignore

        print(f"Name: {name_match}")
        print(f"Params: {params_match}")
        print(f"Instance: {instance_match}")
        print(f"Repetition: {rep_match}")
        data = {}
        if name_match == "LBBDSolver":
            forbidden_vars_regex = r"forbidden_vars:"
            count_forbidden_vars = len(re.findall(forbidden_vars_regex, out_file_str))
            all_cuts = r"d_prime:"
            count_cuts = len(re.findall(all_cuts, out_file_str))
            data["count_cuts"] = count_cuts
            data["count_forbidden_vars"] = count_forbidden_vars

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

        util_data = extract_utilization(out_file_str)
        data.update(util_data)  # type: ignore

        instance_file_name = instance_match.split("/")[-1]

        time_slot_length = int(instance_match.split("/")[-1].split("_")[2][1:])


        if name_match == "CPSolver":
            return

        if name_match == "LBBDSolver":
            params = params_match["subsolver_cls"]
            params_match.pop("subsolver_cls")
            name_match = "LBBDSolver_" + params

        all_solv_par_inst_rep[name_match][hash_dict(params_match)][
            instance_file_name
        ] = data
        all_inst_solv_par_rep[instance_file_name][name_match][
            hash_dict(params_match)
        ] = data

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
            return {
                key: recursive_defaultdict_to_dict(value) for key, value in d.items()
            }
        return d  # Base case: return the value as is if it's not a defaultdict

    # Make data to dict
    all_solv_par_inst_rep = recursive_defaultdict_to_dict(all_solv_par_inst_rep)
    all_inst_solv_par_rep = recursive_defaultdict_to_dict(all_inst_solv_par_rep)

    # Remove unwanted data
    for value in remove_runs:
        all_solv_par_inst_rep = remove_keys_with_name(all_solv_par_inst_rep, value)
        all_inst_solv_par_rep = remove_keys_with_name(all_inst_solv_par_rep, value)

    for value in remove_runs_contains:
        all_solv_par_inst_rep = remove_keys_with_name(
            all_solv_par_inst_rep, value, contains=True
        )
        all_inst_solv_par_rep = remove_keys_with_name(
            all_inst_solv_par_rep, value, contains=True
        )

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
            print(
                f"Instance {instance} has at least {val} solver that found a solution."
            )
            instance_str = "_".join(instance.split("_")[:-1])
            instance_solution_count[instance_str] += 0
        else:
            print(f"Instance {instance} has no solver that found a solution.")
            instance_str = "_".join(instance.split("_")[:-1])
            instance_solution_count[instance_str] += 1

    pprint_dict(instance_solution_count)

    for solv in all_solv_par_inst_rep:
        print(f"Solver: {solv}")
        # Calculate the average solution time and how many instances were solved
        avg_time = 0
        avg_time_n = 0
        num_solved = 0
        for inst in all_solv_par_inst_rep[solv]:
            if "objective_value" in all_solv_par_inst_rep[solv][inst]:
                avg_time += all_solv_par_inst_rep[solv][inst]["total_time"]
                avg_time_n += 1
                num_solved += 1
            else:
                avg_time += 7200
                avg_time_n += 1
        print(f"Average time: {avg_time / avg_time_n}")
        print(f"Number of instances solved: {num_solved}")

    return all_solv_par_inst_rep, all_inst_solv_par_rep


if __name__ == "__main__":
    main()
