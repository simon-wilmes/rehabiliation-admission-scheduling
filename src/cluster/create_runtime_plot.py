from src.cluster.extract_data import main, pprint_dict, set_clipboard


all_solv_par_inst_rep, all_inst_solv_par_rep = main()
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

data = defaultdict(int)


def rec_dd():
    return defaultdict(rec_dd)


succ_sol = defaultdict(int)
for solv in all_solv_par_inst_rep:
    for par in all_solv_par_inst_rep[solv]:
        for inst in all_solv_par_inst_rep[solv][par]:
            if "objective_value" in all_solv_par_inst_rep[solv][par][inst]:
                succ_sol[solv] += 1
print(succ_sol)

best_solver_dict = defaultdict(int)
for inst in all_inst_solv_par_rep:
    for par in all_inst_solv_par_rep[inst]:
        minimum_run_time = 1000000000
        best_solver = None
        for solv in all_inst_solv_par_rep[inst][par]:

            if "total_time" in all_inst_solv_par_rep[inst][par][solv]:
                if (
                    minimum_run_time
                    > all_inst_solv_par_rep[inst][par][solv]["total_time"]
                ):
                    minimum_run_time = all_inst_solv_par_rep[inst][par][solv][
                        "total_time"
                    ]
                    best_solver = solv
        if best_solver is not None:
            best_solver_dict[best_solver] += 1
            print(inst, par, best_solver, minimum_run_time)

print(best_solver_dict)

avg_runtime = {}
avg_n = {}


full_data = {
    r"inst\_8",
    r"inst\_7",
    r"inst\_1\_c",
    r"inst\_1\_a",
    r"inst\_2",
    r"inst\_1",
}


for inst in all_inst_solv_par_rep:
    for param in all_inst_solv_par_rep[inst]:
        for solv in all_inst_solv_par_rep[inst][param]:
            if "objective_value" in all_inst_solv_par_rep[inst][param][solv]:
                data[inst] += 1
                if "LBBD" in solv:
                    print(
                        "ForbiddenVars:",
                        inst in full_data,
                        solv,
                        inst,
                        param,
                        all_inst_solv_par_rep[inst][param][solv][
                            "count_forbidden_vars"
                        ],
                    )
data = dict(data)
cuts_added_solv = defaultdict(int)
avg_subsolver = []
avg_cuts = 0
for inst in all_inst_solv_par_rep:
    if inst not in full_data:
        continue
    for param in all_inst_solv_par_rep[inst]:
        for solv in all_inst_solv_par_rep[inst][param]:
            if not "LBBDSolver" in solv:
                continue
            if all_inst_solv_par_rep[inst][param][solv]["count_forbidden_vars"] > 0:

                print(
                    inst,
                    param,
                    solv,
                    all_inst_solv_par_rep[inst][param][solv]["count_forbidden_vars"],
                )
                avg_subsolver.append(
                    (
                        all_inst_solv_par_rep[inst][param][solv][
                            "subsolver_create_model_time"
                        ]
                        + all_inst_solv_par_rep[inst][param][solv][
                            "subsolver_solve_model_time"
                        ]
                    )
                    / all_inst_solv_par_rep[inst][param][solv]["time_in_solver"]
                )
                print(all_inst_solv_par_rep[inst][param][solv]["count_cuts"])
                avg_cuts += all_inst_solv_par_rep[inst][param][solv]["count_cuts"]
                print(
                    (
                        all_inst_solv_par_rep[inst][param][solv][
                            "subsolver_create_model_time"
                        ]
                        + all_inst_solv_par_rep[inst][param][solv][
                            "subsolver_solve_model_time"
                        ]
                    )
                    / all_inst_solv_par_rep[inst][param][solv]["time_in_solver"]
                )

print("Avg Subsolver", sum(avg_subsolver) / len(avg_subsolver))
print("Avg. Cuts", avg_cuts / len(avg_subsolver))
for inst in all_inst_solv_par_rep:
    if data[inst] != 9:
        continue
    for param in all_inst_solv_par_rep[inst]:
        for solv in all_inst_solv_par_rep[inst][param]:
            if "LBBD" in solv:
                cuts_added_solv[solv] += int(
                    all_inst_solv_par_rep[inst][param][solv]["count_forbidden_vars"]
                    == 0
                )
                if (
                    all_inst_solv_par_rep[inst][param][solv]["count_forbidden_vars"]
                    == 0
                ):
                    pass
                else:
                    print(
                        inst,
                        param,
                        solv,
                        all_inst_solv_par_rep[inst][param][solv][
                            "count_forbidden_vars"
                        ],
                    )

            if "objective_value" in all_inst_solv_par_rep[inst][param][solv]:

                if solv not in avg_n or param not in avg_n[solv]:
                    if solv not in avg_n:
                        avg_n[solv] = {}
                        avg_runtime[solv] = {}
                    avg_n[solv][param] = 0
                    avg_runtime[solv][param] = 0

                avg_n[solv][param] += 1
                avg_runtime[solv][param] += all_inst_solv_par_rep[inst][param][solv][
                    "total_time"
                ]


print(dict(data))
print(avg_n)
print(avg_runtime)
for key in avg_runtime:
    for param in avg_runtime[key]:
        print(key, param, avg_runtime[key][param] / avg_n[key][param])


# Define color and marker mappings
parameters = {param for instance in all_inst_solv_par_rep.values() for param in instance.keys()}  # type: ignore
parameters = sorted(list(parameters))
algorithms = {
    alg
    for instance in all_inst_solv_par_rep.values()
    for param in instance.values()  # type: ignore
    for alg in param.keys()
}
algorithms.remove("MIPSolver3")

param_markers = {
    param: marker for param, marker in zip(parameters, ["o", "s", "^", "D", "P"])
}
alg_colors = {alg: color for alg, color in zip(algorithms, plt.cm.tab10.colors)}  # type: ignore

# Prepare plot
fig, ax = plt.subplots()

# Flattened x-ticks and y-values for scatter
x_labels = sorted(list(full_data))
x_positions = {instance: idx for idx, instance in enumerate(x_labels)}

for instance_name in sorted(all_inst_solv_par_rep.keys()):
    param_dict = all_inst_solv_par_rep[instance_name]
    if instance_name not in full_data:
        continue
    x_pos = x_positions[instance_name]
    for param, alg_dict in param_dict.items():
        for alg, runtime_dict in alg_dict.items():
            runtime = runtime_dict.get("total_time")
            if runtime is not None:
                ax.scatter(
                    x_pos,
                    runtime,
                    color=alg_colors[alg],
                    marker=param_markers[param],
                    label=f"{alg} ({param})",
                )

# Set x-ticks and labels
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=45, ha="right")

# Label axes
ax.set_xlabel("Instance")
ax.set_ylabel("Total Runtime (in seconds)")

# Set maximum value for y-axis
ax.set_ylim(top=600)  # Adjust the value as needed
ax.set_ylim(bottom=0)  # Adjust the value as needed

# Ensure y-axis is integer if runtimes are integers
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Avoid duplicate legends
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))

rename_dict = {
    "LBBDSolver_CPSubSolver (5)": r"LBBD-Resources (5min)",
    "LBBDSolver_CPSubSolver (15)": r"LBBD-Resources (15min)",
    "LBBDSolver_CPSubSolver (30)": r"LBBD-Resources (30min)",
    "MIPSolver (5)": "MIP-RUA (5min)",
    "MIPSolver (15)": "MIP-RUA (15min)",
    "MIPSolver (30)": "MIP-RUA (30min)",
    "LBBDSolver_CPSubSolver2 (5)": r"LBBD-Groupsize (5min)",
    "LBBDSolver_CPSubSolver2 (15)": r"LBBD-Groupsize (15min)",
    "LBBDSolver_CPSubSolver2 (30)": r"LBBD-Groupsize (30min)",
}
new_unique_labels = []
new_unique_handles = []
unique_labels = dict(
    sorted(
        list(unique_labels.items()),
        key=lambda x: (x[0].split("(")[0], int(x[0].split("(")[1].split(")")[0])),
    )
)
for unique_label in unique_labels:
    new_unique_labels.append(rename_dict[unique_label])
    new_unique_handles.append(unique_labels[unique_label])


ax.legend(
    new_unique_handles,
    new_unique_labels,
    title="Legend",
    bbox_to_anchor=(0.5, 1.25),  # Center the legend above the plot
    loc="center",  # Use "center" to center align the legend
    ncol=3,  # Arrange the legend into 3 columns
    handletextpad=0.5,  # Reduce space between marker and text
    columnspacing=0.4,
)

# Adjust layout for better fit
plt.tight_layout()

# Export plot to PGF file
plt.savefig("src/cluster/pgf/scatter_plot.pgf", format="pgf")
with open("src/cluster/pgf/scatter_plot.pgf", "r") as f:
    set_clipboard(f.read())
plt.show()
input()
