try:
    import pyperclip

    pyperclip_available = True
except ImportError:
    pyperclip_available = False

import builtins
from src.utils import calculate_dict_changes, calculate_similarity_scores


def custom_print(*args, file_path="output.log", **kwargs):
    output = " ".join(map(str, args))
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    builtins._original_print(*args, **kwargs)  # type: ignore


# Backup the original print
builtins._original_print = builtins.print  # type: ignore

# Override the built-in print
builtins.print = custom_print
from datetime import datetime, timedelta

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

extract_solve_length = True

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
    / "study_4"
)


def recursive_defaultdict_to_dict(d):
    if isinstance(d, defaultdict):  # Check if the current level is a defaultdict
        return {key: recursive_defaultdict_to_dict(value) for key, value in d.items()}
    return d  # Base case: return the value as is if it's not a defaultdict


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


# Regex pattern to match the timestamp
date_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}"


def find_first_date(out_file_lines):

    for line in out_file_lines:
        match = re.search(date_pattern, line)
        if match:
            return datetime.strptime(
                match.group(0), "%Y-%m-%d %H:%M:%S,%f"
            )  # Return the first match
    return None  # No match found


def parse_log(out_file_str, is_time_out=False):
    out_file_lines = out_file_str.split("\n")
    start_time = None
    time_differences = []
    log_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - DEBUG - ---------- (SOLVE|END) SUBSYSTEM ---------"
    for line in out_file_lines:
        match = re.search(log_pattern, line)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

            if match.group(2) == "SOLVE":  # Start of subsystem
                if start_time is not None:
                    assert False, "Start time already set"
                start_time = timestamp
            elif match.group(2) == "END" and start_time:  # End of subsystem
                time_difference = timestamp - start_time
                time_differences.append(
                    round(
                        time_difference.seconds + time_difference.microseconds / 1e6, 3
                    )
                )
                start_time = None  # Reset for the next cycle

    # Handle unmatched start
    if start_time is not None:
        first_date = find_first_date(out_file_lines)
        if first_date is None:
            raise ValueError("No date found in the log file")
        first_date += timedelta(hours=1)
        new_length = round(
            (first_date - start_time).seconds
            + (first_date - start_time).microseconds / 1e6,
            3,
        )

        if is_time_out and new_length < 3600:
            time_differences.append(new_length)

    return time_differences


def main():
    import pickle

    # Define the file path for the pickle file
    pickle_file = "data.pkl"

    # Check if the pickle file exists
    if os.path.exists(pickle_file):

        print("Object loaded from pickle file.")
        # Load the object from the pickle file
        with open(pickle_file, "rb") as f:
            return pickle.load(f)

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
            "subsolver_create_model_time": round(float(create_model_match), 2),
            "subsolver_solve_model_time": round(float(solve_model_match), 2),
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
            "time_build_model": round(float(time_build_model_match), 2),
            "time_in_solver": round(float(time_in_solver_match), 2),
            "total_time": round(float(total_time_match), 2),
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
        data = {"hash": job_name_match.split("_")[2]}

        if "vigh" in instance_match:
            return

        # extract memory usage if available:
        memory_regex = r"\.batch\s+(\d+)([KMGT])"

        memory_match = re.findall(memory_regex, out_file_str)
        if memory_match:
            memory = int(memory_match[-1][0])
            memory_unit = memory_match[-1][1]
            if memory_unit != "K":
                print(f"Memory unit not recognized: {memory_unit}")
            data["memory_in_GB"] = round(memory / 1024 / 1024, 3)  # in GB

            # count subsolver time calls:

        if test_memory_limit(err_file_str):
            data.update({"memory_limit": True})
        elif test_timeout(err_file_str):
            data.update({"timeout": True})
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

        if name_match == "LBBDSolver":

            results = defaultdict(list)
            if (
                "p30_t30_low" in instance_match
                and params_match["subsolver_cls"] == "CPSubsolver"
            ):
                # Join all lines with newlines to preserve structure

                # Pattern to match both lines together
                # (?s) enables dot to match newlines
                pattern = r"[\d-]+ \d{2}:\d{2}:\d{2},\d{3} - INFO - d_prime: \d+ d: (\d+)\n.*?forbidden_vars: (\[.*?\])"

                # Find all matches in the text
                matches = list(re.finditer(pattern, out_file_str))

                for ind, _ in enumerate(out_file):
                    if "forbidden_vars" in out_file[ind]:
                        day_match = re.findall(
                            r"d_prime: \d+ d: (\d+)", out_file[ind - 1]
                        )
                        vars_match = re.findall(
                            r"forbidden_vars: (\(\[.*?\]\))", out_file[ind]
                        )
                        results[int(day_match[0])].append(eval(vars_match[0]))

                data["forbidden_vars"] = results
                similarity_score = {}

                for day in results:
                    break
                    l1, l2 = calculate_similarity_scores(results[day])
                    l2_sorted = dict(sorted(l2.items()))
                    similarity_score[day] = l2_sorted

                    pass

            forbidden_vars_regex = (
                r"Number of cuts added: (\d+) \(bad=(\d+), good=(\d+)\)"
            )

            count_cuts = re.findall(forbidden_vars_regex, out_file_str)
            if count_cuts:
                good_cuts = int(count_cuts[-1][2])
                bad_cuts = int(count_cuts[-1][1])
                data["good_cuts"] = good_cuts
                data["bad_cuts"] = bad_cuts
            else:
                data["good_cuts"] = 0
                data["bad_cuts"] = 0
            calls_is_day = (
                r"Calls to is_day_infeasible: cp_solver=(\d+)/ total=(\d+) stored=(\d+)"
            )
            count_calls = re.findall(calls_is_day, out_file_str)
            if count_calls:
                data["calls_is_day_infeasible"] = int(count_calls[-1][0])
                data["calls_stored"] = int(count_calls[-1][2])

            solve_submodel_regex = r"Subsolver: Solve Model: (\d+\.\d+)/0\.(\d+)"
            solve_submodel_time = re.findall(solve_submodel_regex, out_file_str)
            if solve_submodel_time:
                data["solve_submodel_time"] = round(
                    float(solve_submodel_time[-1][0]), 3
                )
            if extract_solve_length and name_match == "LBBDSolver":

                times = parse_log(out_file_str, is_time_out="timeout" in data)
                if times:
                    avg_sub_time = sum(times) / len(times)
                    median_sub_time = sorted(times)[len(times) // 2]
                    data["avg_sub_time"] = avg_sub_time
                    data["median_sub_time"] = median_sub_time
                    data["sub_times"] = times

        if name_match == "LBBDSolver":
            params = params_match["subsolver_cls"]
            params_match.pop("subsolver_cls")
            name_match = "LBBDSolver_" + params

        all_solv_par_inst_rep[name_match][hash_dict(params_match)][  # type: ignore
            instance_file_name
        ] = data
        all_inst_solv_par_rep[instance_file_name][name_match][  # type: ignore
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
    for instance, solvers in all_inst_solv_par_rep.items():  # type: ignore
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
    # Save the object to the pickle file
    with open(pickle_file, "wb") as f:
        pickle.dump((all_solv_par_inst_rep, all_inst_solv_par_rep), f)
    return all_solv_par_inst_rep, all_inst_solv_par_rep


if __name__ == "__main__":
    main()
