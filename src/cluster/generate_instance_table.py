from texttable import Texttable
from latextable import draw_latex
import pandas as pd
import pyperclip
import os
from src.instance import create_instance_from_file, Instance
from statistics import median


def set_clipboard(text):
    pyperclip.copy(text)


table = Texttable()

table_data = {
    "instance": [],
    "\#patients": [],
    "\#treatment appointments": [],
}


def calculate_data(instance: Instance):
    data = {}
    data[r"\#patients"] = len(instance.patients)
    data[r"\#treatment appointments"] = sum(
        sum(p.treatments.values()) for p in instance.patients.values()
    )
    data["planning horizon (in days)"] = str(instance.horizon_length)
    data["daily resolution (in min)"] = str(int(instance.time_slot_length.hours * 60))
    data[r"\#resources"] = len(instance.resources)
    data[r"\#resource groups"] = len(instance.resource_groups)
    data[r"\#treatments"] = len(instance.treatments)
    avg_treat = 0
    avg_treat_n = 0
    for p in instance.patients.values():
        avg_treat += sum(p.treatments.values()) / p.length_of_stay
        avg_treat_n += 1
    avg_treat = avg_treat / avg_treat_n
    data[r"avg. \#treatments per day1"] = avg_treat
    data[r"avg. \#treatments per day2"] = sum(
        sum(p.treatments.values()) for p in instance.patients.values()
    ) / sum(p.length_of_stay for p in instance.patients.values())
    return data


frame = pd.DataFrame(table_data)
seen_instances = set()
instance_list = []

replace_words = {
    r"\_less\_prescriptions": r"\_a",
    r"\_more\_resources": r"\_b",
    r"\_short\_timeframe": r"\_c",
}
average_num_treatments = 0
average_num_n = 0
for instance_file in os.listdir("data/comp_study_002"):
    instance_path = "data/comp_study_002/" + instance_file
    instance = create_instance_from_file(instance_path)
    instance_data = calculate_data(instance)
    instance_file_name = r"\_".join(instance_file.split("_")[:-1]).replace(
        r"instance\_", r"inst\_"
    )
    for word in replace_words:
        instance_file_name = instance_file_name.replace(word, replace_words[word])

    if instance_file_name in seen_instances:
        continue
    instance_data["instance"] = instance_file_name
    seen_instances.add(instance_file_name)
    instance_list.append(instance_data)

print("Median", median(inst[r"avg. \#treatments per day2"] for inst in instance_list))

headers = [
    r"instance",
    r"\#patients",
    r"\#treatment appointments",
    "planning horizon (in days)",
    r"\#resources",
]

latex = """
\\begin{table}[t!]
\\begin{tabularx}{\\textwidth}{%COLUMNS}
\\hline
%HEADER
\\hline
%DATA
\\end{tabularx}
\\caption{Key instance characteristics of the instances used in the computational study.}
\\label{tab:instance_characteristics}
\\end{table}
"""
latex = latex.replace("%COLUMNS", "p{2cm}" + "X" * (len(headers) - 1))


latex = latex.replace("%HEADER", "&".join(headers) + "\\\\")
instance_list = sorted(instance_list, key=lambda x: x["instance"])
data_str = ""
for instance_data in instance_list:
    row = []
    for header in headers:
        row.append(str(instance_data.get(header, "")))
    data_str += "&".join(row) + "\\\\" + "\n"

latex = latex.replace("%DATA", data_str)

set_clipboard(latex)
