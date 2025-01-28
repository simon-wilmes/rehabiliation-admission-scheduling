from src.cluster.extract_data import (
    main,
    pprint_dict,
    set_clipboard,
    recursive_defaultdict_to_dict,
)
import src.cluster.extract_data
from matplotlib import rcParams
import numpy as np

all_solv_par_inst_rep, all_inst_solv_par_rep = main()
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import MaxNLocator
from statistics import median
from statistics import geometric_mean


def rec_dd():
    return defaultdict(rec_dd)


succ_sol = defaultdict(int)
for solv in all_solv_par_inst_rep:
    for par in all_solv_par_inst_rep[solv]:
        for inst in all_solv_par_inst_rep[solv][par]:
            if "objective_value" in all_solv_par_inst_rep[solv][par][inst]:
                succ_sol[solv] += 1
print(succ_sol)


# Remove parameters from dicts
all_sol_inst = defaultdict(dict)
for solver in all_solv_par_inst_rep:
    assert len(all_solv_par_inst_rep[solver]) == 1
    for param in all_solv_par_inst_rep[solver]:
        for instance in all_solv_par_inst_rep[solver][param]:
            all_sol_inst[solver][instance] = all_solv_par_inst_rep[solver][param][
                instance
            ]

all_inst_sol = defaultdict(lambda: defaultdict(dict))
for instance in all_inst_solv_par_rep:
    for solver in all_inst_solv_par_rep[instance]:
        for param in all_inst_solv_par_rep[instance][solver]:
            all_inst_sol[instance][solver] = all_inst_solv_par_rep[instance][solver][
                param
            ]

all_sol_inst = recursive_defaultdict_to_dict(all_sol_inst)
all_inst_sol = recursive_defaultdict_to_dict(all_inst_sol)

# pprint_dict(all_sol_inst)
pass
# pprint_dict(all_inst_sol)

pass
avg_runtime_sub_norm = defaultdict(lambda: defaultdict(list))
avg_runtime_sub = defaultdict(lambda: defaultdict(list))
for inst in all_sol_inst["LBBDSolver_CPSubsolver"]:
    patient = inst.split("_")[1]
    time_slot = inst.split("_")[2]
    resource = inst.split("_")[3][:-4]
    data = all_sol_inst["LBBDSolver_CPSubsolver"][inst]
    if "avg_sub_time" in data:
        # calculate normalized time
        d = list(filter(lambda x: x < 10 * data["avg_sub_time"], data["sub_times"]))
        avg_runtime_sub_norm[patient][resource].append(sum(d) / len(d))
        avg_runtime_sub[patient][resource].append(data["avg_sub_time"])


pprint_dict(recursive_defaultdict_to_dict(avg_runtime_sub_norm))
print("Median")
h = "\t\tp10\t\tp20\t\tp30\t\tp40\t\tp50\t\tp60"
print(h)

for time_slot_length in ["low", "medium", "high"]:
    print(time_slot_length, end="\t\t")
    for patient in avg_runtime_sub:
        val = round(
            median(avg_runtime_sub[patient][time_slot_length]),
            4,
        )
        print(val, end="\t")
        if len(str(val)) < 8:
            print("\t", end="")
    print("\n")
print("Geometric")
print(h)
for time_slot_length in ["low", "medium", "high"]:
    print(time_slot_length, end="\t\t")
    for patient in avg_runtime_sub:
        val = round(
            geometric_mean(avg_runtime_sub[patient][time_slot_length]),
            4,
        )
        print(val, end="\t")
        if len(str(val)) < 8:
            print("\t", end="")
    print("\n")
print("Average")
print(h)
for time_slot_length in ["low", "medium", "high"]:
    print(time_slot_length, end="\t\t")
    for patient in avg_runtime_sub:
        val = round(
            sum(avg_runtime_sub[patient][time_slot_length])
            / len(avg_runtime_sub[patient][time_slot_length]),
            4,
        )
        print(val, end="\t")
        if len(str(val)) < 8:
            print("\t", end="")
    print("\n")
cuts_list = {}
type_cut_list = []
time_slot_cuts = defaultdict(list)
resource_cuts = defaultdict(list)
timeout_cuts = {}
memory_out_cuts = {}
patient_cuts = defaultdict(list)
inst_cuts = defaultdict(list)
succ_cuts = {}


for inst in all_sol_inst["LBBDSolver_CPSubsolver"]:
    if "low" not in inst:
        continue
    patient = inst.split("_")[1]
    time_slot = inst.split("_")[2]
    resource = inst.split("_")[3][:-4]
    data = all_sol_inst["LBBDSolver_CPSubsolver"][inst]

    if "good_cuts" not in data:
        continue
    cut_data = (data["good_cuts"], data["bad_cuts"])

    inst_cuts[inst].append(cut_data)
    patient_cuts[patient].append(cut_data)
    type_cut_list.append(cut_data)
    time_slot_cuts[time_slot].append(cut_data)
    resource_cuts[resource].append(cut_data)
    if "timeout" in data:
        timeout_cuts[inst] = cut_data

    elif "memory_limit" in data:
        timeout_cuts[inst] = cut_data
    else:
        succ_cuts[inst] = cut_data

# assert len(all_inst_sol) == len(succ_cuts) + len(timeout_cuts)
# Sample data
instance1 = [x[9:16] for x in sorted(list(succ_cuts.keys()))]
values1 = [succ_cuts[x][0] for x in sorted(list(succ_cuts.keys()))]

instance2 = [x[9:16] for x in sorted(list(timeout_cuts.keys()))]
values2 = [timeout_cuts[x][0] for x in sorted(list(timeout_cuts.keys()))]

# Set figure size for beamer slides (typically 128mm x 96mm)

# First plot
plt.figure(figsize=(3, 4))
plt.bar(instance1, values1)
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig("num_cuts1.pdf", bbox_inches="tight", format="pdf")
plt.close()

# Second plot
plt.figure(figsize=(3, 4))
plt.bar(instance2, values2)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("num_cuts2.pdf", bbox_inches="tight", format="pdf")
plt.close()


pprint_dict(inst_cuts)
pass
pprint_dict(timeout_cuts)
pass
pprint_dict(patient_cuts)
pass
pprint_dict(time_slot_cuts)
pass
pprint_dict(resource_cuts)
pass
pprint_dict(succ_cuts)


solvers = [
    "LBBDSolver_CPSubsolver",
    "LBBDSolver_MIPSubsolver",
    "MIPSolver3",
    "MIPSolver",
]
average_factor = 0
num_factor = 0
list_improvement = []
for inst in all_inst_sol:

    try:
        CP_sol = all_inst_sol[inst]["LBBDSolver_CPSubsolver"]["total_time"]
        try:
            MIP_sol = all_inst_sol[inst]["LBBDSolver_MIPSubsolver"]["total_time"]
            print(CP_sol, MIP_sol, 1 / (CP_sol / MIP_sol))
            list_improvement.append(1 / (CP_sol / MIP_sol))
            average_factor += MIP_sol / CP_sol
            num_factor += 1
        except:
            pass
    except:
        pass

solv_inten_avg = defaultdict(lambda: defaultdict(dict))
solv_inten_median = defaultdict(lambda: defaultdict(dict))
solv_inten_num = defaultdict(lambda: defaultdict(dict))
for inst in all_inst_sol:
    if "t15" not in inst:
        continue
    pass
    inten = inst.split("_")[3][:-4]
    patient = inst.split("_")[1]

    try:
        solv_inten_median[patient][inten]["CPSubsolver"] = all_inst_sol[inst][
            "LBBDSolver_CPSubsolver"
        ]["median_sub_time"]
        solv_inten_num[patient][inten]["CPSubsolver"] = len(
            all_inst_sol[inst]["LBBDSolver_CPSubsolver"]["sub_times"]
        )
    except:
        pass
    try:
        solv_inten_median[patient][inten]["MIPSubsolver"] = all_inst_sol[inst][
            "LBBDSolver_MIPSubsolver"
        ]["median_sub_time"]
        solv_inten_num[patient][inten]["MIPSubsolver"] = len(
            all_inst_sol[inst]["LBBDSolver_MIPSubsolver"]["sub_times"]
        )
    except:
        pass
    try:
        solv_inten_avg[patient][inten]["CPSubsolver"] = all_inst_sol[inst][
            "LBBDSolver_CPSubsolver"
        ]["avg_sub_time"]
    except:
        pass
    try:
        solv_inten_avg[patient][inten]["MIPSubsolver"] = all_inst_sol[inst][
            "LBBDSolver_MIPSubsolver"
        ]["avg_sub_time"]
    except:
        pass


pprint_dict(recursive_defaultdict_to_dict(solv_inten_avg))
pprint_dict(recursive_defaultdict_to_dict(solv_inten_median))
pprint_dict(recursive_defaultdict_to_dict(solv_inten_num))
print(list_improvement)
print(geometric_mean(list_improvement))
print(median(list_improvement))

print(sum(list_improvement) / len(list_improvement))
pass
