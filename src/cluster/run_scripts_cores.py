import os
from itertools import product
from pathlib import Path
from copy import copy
from time import sleep
from random import shuffle
import sys
import hashlib
import subprocess

# 0. Define all parameters

params_all = {
    "$SCRIPT_FOLDER": "/home/wx350715/Kobra/rehabiliation-admission-scheduling",
    "$OUTPUT_FOLDER": "/work/wx350715/Kobra/output_core_test",
    "$RUNTIME": "00:20:00",
    "$MEMORY_PER_CORE": 5000,
    "$PARTITION": "c23mm",  # "c23mm",
}

params_multiple = [
    {
        "$REPETITION": 1,
        "$CPU_CORES": 1,
    },
    {
        "$REPETITION": 1,
        "$CPU_CORES": 4,
    },
    {
        "$REPETITION": 1,
        "$CPU_CORES": 8,
    },
    {
        "$REPETITION": 1,
        "$CPU_CORES": 16,
    },
]
"""
    {
        "$RUNTIME": "00:20:00",
        "$MEMORY": "10000MB",
        "$PARTITION": "c23ml",  # "c23mm",
        "$REPETITION": 1,
        "$CPU_CORES": 20,
    },
    {
        "$RUNTIME": "00:20:00",
        "$MEMORY": "10000MB",
        "$PARTITION": "c23ml",  # "c23mm",
        "$REPETITION": 2,
        "$CPU_CORES": 1,
    },
    {
        "$RUNTIME": "00:20:00",
        "$MEMORY": "10000MB",
        "$PARTITION": "c23ml",  # "c23mm",
        "$REPETITION": 2,
        "$CPU_CORES": 4,
    },
    {
        "$RUNTIME": "00:20:00",
        "$MEMORY": "10000MB",
        "$PARTITION": "c23ml",  # "c23mm",
        "$REPETITION": 2,
        "$CPU_CORES": 8,
    },
    {
        "$RUNTIME": "00:20:00",
        "$MEMORY": "10000MB",
        "$PARTITION": "c23ml",  # "c23mm",
        "$REPETITION": 2,
        "$CPU_CORES": 16,
    },
    {
        "$RUNTIME": "00:20:00",
        "$MEMORY": "10000MB",
        "$PARTITION": "c23ml",  # "c23mm",
        "$REPETITION": 2,
        "$CPU_CORES": 20,
    },
]
"""

TEMPLATE_PATH = "/home/wx350715/Kobra/rehabiliation-admission-scheduling/src/cluster/cluster_cores.template"
SCRIPTS_FOLDER = "/work/wx350715/Kobra/scripts"

# 1. Find all the instance files for the computational_study
INSTANCE_FOLDER_PATH = (
    "/home/wx350715/Kobra/rehabiliation-admission-scheduling/data/core_test_2"
)

# Get all instance files
assert os.path.isdir(INSTANCE_FOLDER_PATH)


# 2. Generate all the combinations to run for the solvers
all_solver_combis = [
    {
        "solver": "MIPSolver",
        "use_lazy_constraints": False,
        "number_of_threads": 1,
    },
    {
        "solver": "MIPSolver",
        "use_lazy_constraints": False,
        "number_of_threads": 4,
    },
    {
        "solver": "MIPSolver",
        "use_lazy_constraints": False,
        "number_of_threads": 8,
    },
    {
        "solver": "MIPSolver",
        "use_lazy_constraints": False,
        "number_of_threads": 16,
    },
    {
        "solver": "LBBDSolver",
        "break_symmetry": True,
        "add_constraints_to_symmetric_days": True,
        "subsolver_cls": "CPSubsolver",
        "number_of_threads": 1,
    },
    {
        "solver": "LBBDSolver",
        "break_symmetry": True,
        "add_constraints_to_symmetric_days": True,
        "subsolver_cls": "CPSubsolver",
        "number_of_threads": 4,
    },
    {
        "solver": "LBBDSolver",
        "break_symmetry": True,
        "add_constraints_to_symmetric_days": True,
        "subsolver_cls": "CPSubsolver",
        "number_of_threads": 8,
    },
    {
        "solver": "LBBDSolver",
        "break_symmetry": True,
        "add_constraints_to_symmetric_days": True,
        "subsolver_cls": "CPSubsolver",
        "number_of_threads": 16,
    },
]
instance_files = [
    os.path.join(INSTANCE_FOLDER_PATH, f)
    for f in os.listdir(INSTANCE_FOLDER_PATH)
    if os.path.isfile(os.path.join(INSTANCE_FOLDER_PATH, f))
]
print(f"Found {len(instance_files)} instance files.")

# 3. Read in the slurm template
with open(TEMPLATE_PATH, "r") as f:
    template = f.read()

# 4. Check with combinations are still missing
output_path = params_all["$OUTPUT_FOLDER"]

# Get all the output files in the output path directory
assert os.path.isdir(output_path)
all_files_output = [
    Path(output_path) / f
    for f in os.listdir(output_path)
    if os.path.isfile(os.path.join(output_path, f))
]

# Get the starting hash from the output files
hashes = set()
for output_file in all_files_output:
    hash = output_file.name.split("_")[2]
    hashes.add(hash)

to_run_combis = []
assert os.path.isdir(params_all["$OUTPUT_FOLDER"])
assert os.path.isdir(params_all["$SCRIPT_FOLDER"])

core_hours_required = 0
for solver_combi, params_combi, instance_file in product(
    all_solver_combis, params_multiple, instance_files
):
    # get output hash
    hash_copy = copy(solver_combi)
    for param_key in params_combi:
        hash_copy[param_key] = params_combi[param_key]

    hash_combi_str = str(
        sorted(
            [(key, value) for key, value in hash_copy.items()],
            key=lambda x: x[0],
        )
    )
    hash_combi_str = str(hash_combi_str + instance_file).encode()
    hash = hashlib.md5(hash_combi_str).hexdigest()
    params_combi_copy = copy(params_combi)
    for key in params_all:
        params_combi_copy[key] = params_all[key]
    params_combi_copy["$MEMORY"] = (
        str(
            max(
                params_combi_copy["$MEMORY_PER_CORE"]
                * int(params_combi_copy["$CPU_CORES"]),
                5000,
            )
        )
        + "MB"
    )

    if hash in hashes:
        print(".", end="")
    else:
        print("x", end="")
        to_run_combis.append((solver_combi, params_combi_copy, instance_file, hash))
        runtime_in_hours = params_combi_copy["$RUNTIME"].split(":")
        try:
            runtime_in_hours = int(runtime_in_hours[0]) + int(runtime_in_hours[1]) / 60
            core_hours_required += runtime_in_hours * params_combi_copy["$CPU_CORES"]
        except:
            print(
                f"Error with runtime: {params_combi_copy['$RUNTIME']}. Will be Ignored!"
            )

print("")

print(f"Need to run {len(to_run_combis)} combinations.")
print(f"Using {core_hours_required:.1f} core hours.")
if len(sys.argv) < 2 or sys.argv[1] != "-y":
    input("Press Enter to continue...")
    pass

to_run_scripts = []
# 5. Create the slurm scripts
for combi in to_run_combis:
    # Create the template by replacing the parameters
    solver_combi, params_combi, instance_file, hash = combi
    params_copy = copy(params_combi)
    # Update dictionary
    params_copy["$HASH"] = hash
    params_copy["$SOLVER_INSTANCE"] = instance_file
    params_copy["$SOLVER_NAME"] = solver_combi["solver"]

    hash_copy = copy(solver_combi)
    hash_copy.pop("solver")
    params_copy["$SOLVER_PARAMS"] = hash_copy

    print("Create template with the following params and hash:")
    print(params_copy)
    print(hash)

    template_copy = copy(template)
    for key, value in params_copy.items():
        template_copy = template_copy.replace(key, str(value))

    # assert "$" not in template_copy, "Not all parameters were replaced."
    # Write the template to a file
    output_file = Path(SCRIPTS_FOLDER) / f"{hash}_cluster.sh"
    to_run_scripts.append(output_file)

    with open(output_file, "w+") as f:
        f.write(template_copy)

print("Finished creating scripts. Now starting the slurm jobs.")
# Get the number of currently submitted jobs
result = subprocess.run(["squeue", "-u", os.getlogin()], stdout=subprocess.PIPE)
current_jobs = (
    len(result.stdout.decode("utf-8").strip().split("\n")) - 1
)  # subtract 1 for the header line

# Define the maximum number of jobs you can run at the same time
max_jobs = 100
max_jobs_to_run = 100
print("Currently running jobs:", current_jobs)
print("Maximum number of jobs:", max_jobs)
if max_jobs - current_jobs <= 0:
    print("To many jobs are already running. Exiting.")
    exit(0)
jobs_to_start = min(max_jobs - current_jobs, len(to_run_scripts))
print(
    "Number possible number of jobs to start:",
    jobs_to_start,
)
if max_jobs_to_run < jobs_to_start:
    print("Restricting the number of jobs to start to", max_jobs_to_run)

sleep(1)
# Calculate the number of jobs to start


# Shuffle the scripts so that the first k jobs are as diverse as possible
shuffle(to_run_scripts)
count = 0
# Start the jobs
result = subprocess.run(["squeue", "--me", "-o '%.100j'"], stdout=subprocess.PIPE)
squeue_output = result.stdout.decode("utf-8")

for i, script in enumerate(to_run_scripts):
    name_of_script = str(script).split("/")[-1].split("_")[0]
    if name_of_script in squeue_output:
        print(f"Job {i} {script} has already been submitted.")
        continue
    print(f"Submitting job {i} {script}.")
    subprocess.run(["sbatch", script])

    sleep(0.2)
    count += 1
    if count >= min(max_jobs_to_run, jobs_to_start):
        break

print(f"Successfully submitted {count} jobs.")
