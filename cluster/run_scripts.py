import os
from itertools import product
from pathlib import Path

# 1. Find all the instance files for the computational_study
folder_path = "/home/simon/Documents/uni/Kopra/rehabiliation-admission-scheduling/data/comp_study_001"

# Get all instance files
assert os.path.isdir(folder_path)

instance_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]
print(f"Found {len(instance_files)} instance files.")


# 2. Generate all the combinations to run for the solvers

all_combis = [
    ("CPSolver", {"break_symmetry": True}),
    ("CPSolver", {"break_symmetry": False}),
    ("MIPSolver", {"use_lazy_constraints": True}),
    ("MIPSolver", {"use_lazy_constraints": False}),
    ("MIPSolver3", {"break_symmetry": False, "break_symmetry_strong": False}),
    ("MIPSolver3", {"break_symmetry": True, "break_symmetry_strong": False}),
    ("MIPSolver3", {"break_symmetry": True, "break_symmetry_strong": True}),
    (
        "LBBDSolver",
        {
            "break_symmetry": False,
            "add_constraints_to_symmetrc_days": False,
            "subsolver_cls": "CPSubSolver",
        },
        "LBBDSolver",
        {
            "break_symmetry": True,
            "add_constraints_to_symmetrc_days": False,
            "subsolver_cls": "CPSubSolver",
        },
        "LBBDSolver",
        {
            "break_symmetry": False,
            "add_constraints_to_symmetrc_days": True,
            "subsolver_cls": "CPSubSolver",
        },
        "LBBDSolver",
        {
            "break_symmetry": True,
            "add_constraints_to_symmetrc_days": True,
            "subsolver_cls": "CPSubSolver",
        },
        "LBBDSolver",
        {
            "break_symmetry": False,
            "add_constraints_to_symmetrc_days": False,
            "subsolver_cls": "CPSubSolver2",
        },
        "LBBDSolver",
        {
            "break_symmetry": True,
            "add_constraints_to_symmetrc_days": False,
            "subsolver_cls": "CPSubSolver2",
        },
        "LBBDSolver",
        {
            "break_symmetry": False,
            "add_constraints_to_symmetrc_days": True,
            "subsolver_cls": "CPSubSolver2",
        },
        "LBBDSolver",
        {
            "break_symmetry": True,
            "add_constraints_to_symmetrc_days": True,
            "subsolver_cls": "CPSubSolver2",
        },
    ),
]


# 3. Read in the slurm template
template_path = "todo"
with open(template_path, "r") as f:
    template = f.read()


# 4. Check with combinations are still missing
output_path = "todo"

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
    hash = output_file.name.split("-")[0]
    hashes.add(hash)

import hashlib

to_run_combis = []

for solver_combi, instance_file in product(all_combis, instance_files):
    # get output hash
    combi_str = str((solver_combi, instance_file)).encode()
    hash = hashlib.md5(combi_str).hexdigest()

    if hash in hashes:
        print(".")
    else:
        print("x")
        to_run_combis.append((solver_combi, instance_file, hash))

print(f"Need to run {len(to_run_combis)} combinations.")
