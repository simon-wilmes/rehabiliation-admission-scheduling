import os
from itertools import product
from pathlib import Path
from copy import copy
from time import sleep
from random import shuffle

# 0. Define all parameters
params = {
    "$RUNTIME": "01:00:00",
    "$MEMORY": "15360MB",
    "$PARTITION": "c23ml",  # "c23mm",
    "$OUTPUT_FOLDER": "/work/wx350715/Kobra/output",
    "$SCRIPT_FOLDER": "/home/wx350715/Kobra/rehabiliation-admission-scheduling",
    "$REPETITION": 1,
}
params_for_hash = {"$PARTITION", "$MEMORY", "$RUNTIME"}

TEMPLATE_PATH = "cluster/cluster.template"
SCRIPTS_FOLDER = "/work/wx350715/Kobra/scripts"

# 1. Find all the instance files for the computational_study
folder_path = (
    "/home/wx350715/Kobra/rehabiliation-admission-scheduling/data/comp_study_002"
)

# Get all instance files
assert os.path.isdir(folder_path)
assert os.path.isdir(params["$OUTPUT_FOLDER"])
assert os.path.isdir(params["$SCRIPT_FOLDER"])
assert os.path.isfile(params["$SCRIPT_FOLDER"] + "/" + TEMPLATE_PATH)

instance_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]
print(f"Found {len(instance_files)} instance files.")


# 2. Generate all the combinations to run for the solvers

all_combis = [
    {"solver": "CPSolver", "break_symmetry": True},
    {"solver": "CPSolver", "break_symmetry": False},
    {"solver": "MIPSolver", "use_lazy_constraints": True},
    {"solver": "MIPSolver", "use_lazy_constraints": False},
    {"solver": "MIPSolver3", "break_symmetry": False, "break_symmetry_strong": False},
    {"solver": "MIPSolver3", "break_symmetry": True, "break_symmetry_strong": False},
    {"solver": "MIPSolver3", "break_symmetry": True, "break_symmetry_strong": True},
    {
        "solver": "LBBDSolver",
        "break_symmetry": False,
        "add_constraints_to_symmetrc_days": False,
        "subsolver_cls": "CPSubSolver",
    },
    {
        "solver": "LBBDSolver",
        "break_symmetry": True,
        "add_constraints_to_symmetrc_days": False,
        "subsolver_cls": "CPSubSolver",
    },
    {
        "solver": "LBBDSolver",
        "break_symmetry": False,
        "add_constraints_to_symmetrc_days": True,
        "subsolver_cls": "CPSubSolver",
    },
    {
        "solver": "LBBDSolver",
        "break_symmetry": True,
        "add_constraints_to_symmetrc_days": True,
        "subsolver_cls": "CPSubSolver",
    },
    {
        "solver": "LBBDSolver",
        "break_symmetry": False,
        "add_constraints_to_symmetrc_days": False,
        "subsolver_cls": "CPSubSolver2",
    },
    {
        "solver": "LBBDSolver",
        "break_symmetry": True,
        "add_constraints_to_symmetrc_days": False,
        "subsolver_cls": "CPSubSolver2",
    },
    {
        "solver": "LBBDSolver",
        "break_symmetry": False,
        "add_constraints_to_symmetrc_days": True,
        "subsolver_cls": "CPSubSolver2",
    },
    {
        "solver": "LBBDSolver",
        "break_symmetry": True,
        "add_constraints_to_symmetrc_days": True,
        "subsolver_cls": "CPSubSolver2",
    },
]


# 3. Read in the slurm template
template_path = Path(params["$SCRIPT_FOLDER"]) / TEMPLATE_PATH
with open(template_path, "r") as f:
    template = f.read()


# 4. Check with combinations are still missing
output_path = params["$OUTPUT_FOLDER"]

# Get all the output files in the output path directory
assert os.path.isdir(output_path)
all_output = [
    Path(output_path) / f
    for f in os.listdir(output_path)
    if os.path.isfile(os.path.join(output_path, f))
]

# Get the starting hash from the output files
hashes = set()
for output_file in all_output:
    hash = output_file.name.split("_")[2]
    hashes.add(hash)
import hashlib
import subprocess


to_run_combis = []

for solver_combi, instance_file in product(all_combis, instance_files):
    # get output hash
    solver_combi_copy = copy(solver_combi)
    for param in params_for_hash:
        solver_combi_copy[param] = params[param]

    combi_str = str(
        sorted(
            [(key, value) for key, value in solver_combi_copy.items()],
            key=lambda x: x[0],
        )
    )
    combi_str = str(combi_str + instance_file).encode()
    hash = hashlib.md5(combi_str).hexdigest()
    # print(hash)

    if hash in hashes:
        print(".", end="")
    else:
        print("x", end="")
        to_run_combis.append((solver_combi, instance_file, hash))

print("")

print(f"Need to run {len(to_run_combis)} combinations.")
input("Press Enter to continue...")

to_run_scripts = []
# 5. Create the slurm scripts
for combi in to_run_combis:
    # Create the template by replacing the parameters
    solver_combi, instance_file, hash = combi
    params_copy = copy(params)
    # Update dictionary
    params_copy["$HASH"] = hash
    params_copy["$SOLVER_INSTANCE"] = instance_file
    params_copy["$SOLVER_NAME"] = solver_combi["solver"]

    solver_combi_copy = copy(solver_combi)
    solver_combi_copy.pop("solver")
    params_copy["$SOLVER_PARAMS"] = solver_combi_copy

    print("Create template with the following params and hash:")
    print(params_copy)
    print(hash)

    template_copy = template
    for key, value in params_copy.items():
        template_copy = template_copy.replace(key, str(value))

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
max_jobs_to_run = 10000
print("Currently running jobs:", current_jobs)
print("Maximum number of jobs:", max_jobs)
if max_jobs - current_jobs <= 0:
    print("To many jobs are already running. Exiting.")
    exit(0)


print("Number of jobs to start:", min(max_jobs - current_jobs, len(to_run_scripts)))
sleep(1)
# Calculate the number of jobs to start
jobs_to_start = min(max_jobs - current_jobs, len(to_run_scripts))

# Shuffle the scripts so that the first k jobs are as diverse as possible
shuffle(to_run_scripts)
count = 0
# Start the jobs
for i, script in enumerate(to_run_scripts[:jobs_to_start]):
    print(f"Submitting job {i} {script}.")
    subprocess.run(["sbatch", script])
    sleep(0.4)
    count += 1
    if count >= max_jobs_to_run:
        break

print(f"Successfully submitted {jobs_to_start} jobs.")
