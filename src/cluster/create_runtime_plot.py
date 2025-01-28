from src.cluster.extract_data import (
    main,
    pprint_dict,
    set_clipboard,
    recursive_defaultdict_to_dict,
)
import src.cluster.extract_data
from matplotlib import rcParams
import numpy as np
from matplotlib import patches

all_solv_par_inst_rep, all_inst_solv_par_rep = main()
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d


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

solvers = [
    "LBBDSolver_CPSubsolver",
    "LBBDSolver_MIPSubsolver",
    "MIPSolver3",
    "MIPSolver",
]

replace_solver_names = {
    "LBBDSolver_CPSubsolver": "LBBD-CP",
    "LBBDSolver_MIPSubsolver": "LBBD-IP",
    "MIPSolver3": "IP-EGA",
    "MIPSolver": "IP-RUA",
}


solver_colors = dict(zip(solvers, rcParams["axes.prop_cycle"].by_key()["color"]))

#########################################################
# Count number of solved instances for each solver
#########################################################

count_sol = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
for solver in all_solv_par_inst_rep:
    for param in all_solv_par_inst_rep[solver]:
        for instance in all_solv_par_inst_rep[solver][param]:
            instance_type = instance.split("_")[-1][:-4]
            if "t15" not in instance:
                continue

            if "objective_value" in all_solv_par_inst_rep[solver][param][instance]:
                count_sol[instance_type][solver][0] += 1
            if "timeout" in all_solv_par_inst_rep[solver][param][instance]:
                count_sol[instance_type][solver][1] += 1
            if "memory_limit" in all_solv_par_inst_rep[solver][param][instance]:
                count_sol[instance_type][solver][2] += 1

if False:

    # Get unique categories and solvers
    categories = ["low", "medium", "high"]

    # Define patterns for the stacked bars
    patterns = [
        "",
        "///",
        "ooo",
    ]  # First is solid, second is forward slash, third is backslash

    # Get unique categories and solvers
    solvers = sorted(set(solver for cat in count_sol.values() for solver in cat.keys()))

    # Number of bars per group
    n_solvers = len(solvers)

    # Set the width of each bar and positions of the bars
    width = 0.35
    spacing = 2
    x = np.arange(0, len(categories) * spacing, spacing)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Plot stacked bars for each solver
    for i, solver in enumerate(solvers):
        position = x + width * (i - n_solvers / 2 + 0.5)
        bottom = np.zeros(len(categories))

        # Plot each stack level
        for level in range(3):  # Three levels of stacking
            values = [count_sol[cat][solver][level] for cat in categories]
            if level == 0:
                bars = ax.bar(
                    position,
                    values,
                    width,
                    bottom=bottom,
                    label=solver,
                    color=solver_colors[solver],
                    hatch=patterns[level],
                    fill=True,
                )
                bottom += values
                break
            else:
                bars = ax.bar(
                    position,
                    values,
                    width,
                    bottom=bottom,
                    color="white",  # Set background to white
                    edgecolor=solver_colors[solver],  # Set edge color to solver color
                    hatch=patterns[level],
                    fill=False,
                )
                bottom += values

    # Customize the plot
    ax.grid(True, linestyle="-", alpha=0.7)  # Added grid with dashed lines
    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    # Modify legend to show only solver names (not levels)
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = handles[:n_solvers]
    unique_labels = labels[:n_solvers]
    for i in range(n_solvers):
        for key, value in replace_solver_names.items():
            unique_labels[i] = unique_labels[i].replace(key, value)
    ax.legend(unique_handles, unique_labels)

    # Customize the plot
    ax.set_ylabel("Number of Solved Instances")
    # Set grid to be behind the bars
    ax.set_axisbelow(True)  # Added to ensure grid is behind the bars
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig("treatment_intensity.pgf")
    with open("treatment_intensity.pgf", "r") as f:
        set_clipboard(f.read())

    plt.show()
    pass


def calc_area(y):
    return np.trapz(y, dx=1 / len(y))


#########################################################
# Plot runtime subroutine distribution
#########################################################


if True:

    n_ribbons = 21
    # DATA is simply the list
    runtimes_lists = []
    runtimes_list_succ = []
    runtimes_list_fail = []

    for inst in all_inst_sol:
        solv = "LBBDSolver_CPSubsolver"
        if solv not in all_inst_sol[inst]:
            assert False
        if "sub_times" not in all_inst_sol[inst][solv]:
            continue
        if "total_time" not in all_inst_sol[inst][solv]:
            runtimes_list_fail.append(all_inst_sol[inst][solv]["sub_times"])
        else:
            runtimes_list_succ.append(all_inst_sol[inst][solv]["sub_times"])
        runtimes_lists.append(all_inst_sol[inst][solv]["sub_times"])

    distributions = []
    distributions_succ = []
    distributions_fail = []
    average_area_cum = 0
    average_area_cum_succ = 0
    average_area_cum_fail = 0

    k = 10
    worst_k_ind = []

    for in_list in runtimes_lists:
        in_list = sorted(in_list)
        s = sum(in_list)
        cum_list = [sum(in_list[:i]) / s for i in range(len(in_list))]
        cum_list.append(1)
        distributions.append(cum_list)

        area = calc_area(cum_list)
        average_area_cum += area
        print(area)
        if len(worst_k_ind) < k:
            worst_k_ind.append((area, in_list))
            worst_k_ind = sorted(worst_k_ind, key=lambda x: x[0])
        else:
            if worst_k_ind[-1][0] > area:
                worst_k_ind[-1] = (area, in_list)
                worst_k_ind = sorted(worst_k_ind, key=lambda x: x[0])

    x_common = np.linspace(0, 1, 1000)
    for in_list in runtimes_list_succ:
        in_list = sorted(in_list)
        s = sum(in_list)
        cum_list = [sum(in_list[:i]) / s for i in range(len(in_list))]
        cum_list.append(1)
        distributions_succ.append(cum_list)
        average_area_cum_succ += calc_area(cum_list)

    plt.figure(figsize=(6.5, 4))
    for area, run in worst_k_ind:
        plt.plot([i / (len(run) - 1) for i in range(len(run))], run, color="blue")
    plt.plot(x_common, x_common, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    plt.xlabel("Percentage of Subroutine Calls")
    plt.ylabel("Cumulative runtime of Subroutine")
    plt.title(f"Distribution of Subroutine Runtimes")
    plt.grid(True, alpha=0.3)
    plt.xticks([i / 10 for i in range(11)])
    plt.yticks([i / 10 for i in range(11)])
    # Add legend
    # plt.legend(
    #     handles=legend_handles,
    #     title="Percentile Ranges",
    #     loc="upper right",
    #     bbox_to_anchor=(0.15, 1),  # Place the legend outside the plot
    #     borderaxespad=0,
    #     fontsize="small",  # Adjust font size for readability
    # )
    plt.savefig("cut_runtime_worst.pdf", bbox_inches="tight", format="pdf")
    plt.show()
    plt.close()

    for in_list in runtimes_list_fail:
        in_list = sorted(in_list)
        s = sum(in_list)
        cum_list = [sum(in_list[:i]) / s for i in range(len(in_list))]
        cum_list.append(1)
        distributions_fail.append(cum_list)
        average_area_cum_fail += calc_area(cum_list)

    print("All", round(average_area_cum / len(runtimes_lists) * 2, 3))
    print("Succ", average_area_cum_succ / len(runtimes_list_succ) * 2)
    print("Fail", average_area_cum_fail / len(runtimes_list_fail) * 2)

    # Interpolate all distributions to this common x-axis
    interpolated_values = []
    for dist in distributions:
        x_orig = np.linspace(0, 1, len(dist))  # Original x-values
        interp = np.interp(x_common, x_orig, dist)
        interpolated_values.append(interp)

    # Convert to numpy array for easier manipulation
    interpolated_values = np.array(interpolated_values)

    # Calculate percentile lines
    percentiles = np.linspace(0, 100, n_ribbons)
    percentile_lines = np.percentile(interpolated_values, percentiles, axis=0)

    percentile_lines[0] = np.array([0] * len(x_common))
    # Create plot
    plt.figure(figsize=(6.5, 4))

    # Plot filled regions between percentile lines
    # Store legend entries
    legend_handles = []
    for i in range(len(percentiles) - 1):
        color = plt.cm.viridis((percentiles[i]) / 100)  # Average of two percentiles
        plt.fill_between(
            x_common,
            percentile_lines[i],
            percentile_lines[i + 1],
            color=color,
            alpha=0.7,
        )
        # Add a patch for the legend
        legend_handles.append(
            patches.Patch(
                color=color,
                label=f"{percentiles[i]:.1f}%-{percentiles[i + 1]:.1f}%",
            )
        )
    # Plot the percentile lines themselves
    for i, line in enumerate(percentile_lines):

        plt.plot(x_common, line, color="black", linewidth=0.5, alpha=0.5)
    plt.plot(x_common, x_common, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    plt.xlabel("Percentage of Subroutine Calls")
    plt.ylabel("Cumulative runtime of Subroutine")
    plt.title(f"Distribution of Subroutine Runtimes")
    plt.grid(True, alpha=0.3)
    plt.xticks([i / 10 for i in range(11)])
    plt.yticks([i / 10 for i in range(11)])
    # Add legend
    # plt.legend(
    #     handles=legend_handles,
    #     title="Percentile Ranges",
    #     loc="upper right",
    #     bbox_to_anchor=(0.15, 1),  # Place the legend outside the plot
    #     borderaxespad=0,
    #     fontsize="small",  # Adjust font size for readability
    # )
    plt.savefig("cut_runtime_distribution.pdf", bbox_inches="tight", format="pdf")
    plt.show()
    plt.close()
    plt.figure(figsize=(6.5, 4))

    # Plot filled regions between percentile lines
    # Store legend entries
    legend_handles = []
    for i in range(len(percentiles) - 1):
        color = plt.cm.viridis((percentiles[i]) / 100)  # Average of two percentiles
        plt.fill_between(
            x_common,
            percentile_lines[i],
            percentile_lines[i + 1],
            color=color,
            alpha=0.7,
        )
        # Add a patch for the legend
        legend_handles.append(
            patches.Patch(
                color=color,
                label=f"{percentiles[i]:.1f}%-{percentiles[i + 1]:.1f}%",
            )
        )
    # Plot the percentile lines themselves
    for i, line in enumerate(percentile_lines):
        plt.plot(x_common, line, color="black", linewidth=0.5, alpha=0.5)

    plt.plot(x_common, x_common, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    plt.xlabel("Percentage of Subroutine Calls")
    plt.ylabel("Cumulative runtime of Subroutine")
    # plt.title(f"Distribution of Subroutine Runtimes")
    plt.grid(True, alpha=0.3)
    plt.xticks([i / 10 for i in range(11)])
    plt.yticks([i / 10 for i in range(11)])
    # Add legend
    # plt.legend(
    #     handles=legend_handles,
    #     title="Percentile Ranges",
    #     loc="upper right",
    #     bbox_to_anchor=(0.15, 1),  # Place the legend outside the plot
    #     borderaxespad=0,
    #     fontsize="small",  # Adjust font size for readability
    # )
    plt.savefig(
        "cut_runtime_distribution_bottom.pdf", bbox_inches="tight", format="pdf"
    )
    plt.show()
    pass
#########################################################
# Plot runtime for low instances
#########################################################

# build data dict
data = {}
for instance_name in all_inst_solv_par_rep:
    if "low" not in instance_name:
        continue
    _, instance, version, _ = instance_name.split("_")

    if instance not in data:
        data[instance] = {}
    # get version
    data[instance][version] = {}
    for solver in all_inst_solv_par_rep[instance_name]:
        if "total_time" in all_inst_sol[instance_name][solver]:
            data[instance][version][solver] = all_inst_sol[instance_name][solver][
                "total_time"
            ]
        else:
            data[instance][version][solver] = None
if False:
    use_transparency = True
    # Get unique solvers and versions
    versions = ["t30", "t15", "t05"]
    instances = ["p10", "p20", "p30", "p40", "p50", "p60"]
    # Define marker styles for each solver
    marker_styles = dict(
        zip(solvers, ["o", "^", "X", "s", "D", "v", "P", "*"])
    )  # circle, triangle, square, diamond, inverted triangle, plus, star, x

    # Calculate positions for grouped points
    n_solvers = len(solvers)
    n_versions = len(versions)
    version_spacing = 4
    instance_spacing = 3
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6.5, 4))
    solver_name = {}
    # Plot points for each solver with different markers
    for solver_idx, solver in enumerate(solvers):
        if solver not in ["LBBDSolver_CPSubsolver", "MIPSolver3"]:
            pass
        for version_idx, version in enumerate(versions):
            x_positions = []
            y_values = []
            for instance_idx, instance in enumerate(instances):
                x_base = (
                    instance_idx * (n_versions * version_spacing + instance_spacing)
                    + version_idx * version_spacing
                )
                x_positions.append(x_base)
                y_values.append(data[instance][version][solver])
            # Set transparency if enabled
            alpha = 0.7 if use_transparency else 1.0
            if all([val is None for val in y_values]):
                continue
            ax.scatter(
                x_positions,
                y_values,
                label=(
                    f"{replace_solver_names[solver]}" if version_idx == 0 else ""
                ),  # Only add label for first version
                marker=marker_styles[solver],
                color=solver_colors[solver],
                alpha=alpha,
                s=40,
            )
    # Create ticks for all versions
    all_ticks = []
    all_labels = []
    instance_positions = []
    for i, instance in enumerate(instances):
        base_pos = i * (n_versions * version_spacing + instance_spacing)
        instance_positions.append(
            base_pos + (n_versions * version_spacing - version_spacing) / 2
        )
        for j, version in enumerate(versions):
            all_ticks.append(base_pos + j * version_spacing)
            all_labels.append(version[1:])
    # Set version labels
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels)

    # Set y-axis to log scale
    ax.set_yscale("log")

    # Add instance labels below version names with adjusted positioning for log scale
    ymin, ymax = ax.get_ylim()
    label_offset = 10 ** (np.log10(ymin) - 0.1 * (np.log10(ymax) - np.log10(ymin)))
    for i, instance in enumerate(instances):
        ax.text(
            instance_positions[i],
            label_offset,
            instance,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    # Position the legend on top of the graphic outside of the box
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,  # Number of columns in the legend
        fontsize="small",
        borderpad=0.5,
    )
    plt.subplots_adjust(top=0.8)
    plt.ylabel("Runtime (in seconds)")
    plt.grid(True, alpha=0.3)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("runtime_comparison_full.pdf", bbox_inches="tight", format="pdf")

    plt.show()
    pass
    # FINISHED


if False:

    # Sample data structure
    data = {
        "Instance 1": {
            "v1": {"stage1": 2, "stage2": 3, "stage3": 4, "stage4": 1},
            "v2": {"stage1": 3, "stage2": 2, "stage3": 3, "stage4": 2},
            "v3": {"stage1": 1, "stage2": 4, "stage3": 2, "stage4": 3},
        },
        "Instance 2": {
            "v1": {"stage1": 300, "stage2": 200, "stage3": 300, "stage4": 200},
            "v2": {"stage1": 200, "stage2": 400, "stage3": 200, "stage4": 100},
            "v3": {"stage1": 400, "stage2": 100, "stage3": 300, "stage4": 200},
        },
        "Instance 3": {
            "v1": {"stage1": 0.1, "stage2": 0.3, "stage3": 0.2, "stage4": 0.4},
            "v2": {"stage1": 0.2, "stage2": 0.2, "stage3": 0.4, "stage4": 0.2},
            "v3": {"stage1": 0.3, "stage2": 0.3, "stage3": 0.1, "stage4": 0.3},
        },
        "Instance 4": {
            "v1": {"stage1": 20, "stage2": 10, "stage3": 30, "stage4": 20},
            "v2": {"stage1": 30, "stage2": 30, "stage3": 20, "stage4": 40},
            "v3": {"stage1": 10, "stage2": 20, "stage3": 40, "stage4": 10},
        },
    }

    # Setup plot parameters
    instances = list(data.keys())
    versions = list(data[instances[0]].keys())
    stages = list(data[instances[0]][versions[0]].keys())

    n_versions = len(versions)
    bar_width = 0.25

    # Create figure with subplots
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 6))
    plt.subplots_adjust(wspace=0.3)  # Adjust spacing between subplots

    # Get default color cycle from matplotlib
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(stages)]
    stage_colors = dict(zip(stages, colors))

    # Create legend handles and labels
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color) for color in stage_colors.values()
    ]
    legend_labels = list(stages)

    # Plot each instance in a separate subplot
    for i, (instance, ax) in enumerate(zip(instances, axs)):
        positions = np.arange(n_versions)

        # Plot bars for each version
        for v, version in enumerate(versions):
            bottom = 0
            for stage in stages:
                height = data[instance][version][stage]
                ax.bar(
                    v, height, bar_width * 2, bottom=bottom, color=stage_colors[stage]
                )
                bottom += height

        # Customize this subplot
        ax.set_ylabel("Runtime (s)" if i == 0 else "")
        # Add instance name below plot
        # Add instance name below plot
        ax.text(
            n_versions / 2 - 0.5,
            ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15,
            instance,
            ha="center",
            va="top",
            transform=ax.transData,
        )

        # Set version positions and labels
        ax.set_xticks(positions)
        ax.set_xticklabels(versions)

    # Add single legend for the entire figure
    # Add single legend for the entire figure
    legend = fig.legend(
        legend_patches,
        legend_labels,
        title="Stages",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(stages),
        fontsize="small",
        borderpad=0.5,
    )
    # Set a common super title
    fig.suptitle("Runtime Comparison Across Instances and Versions", y=1.05)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()

pass
