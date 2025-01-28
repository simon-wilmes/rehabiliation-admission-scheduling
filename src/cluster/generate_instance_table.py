from texttable import Texttable
from latextable import draw_latex
import pandas as pd
import pyperclip
import os
from src.instance import create_instance_from_file, Instance
from statistics import median
from copy import copy


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
    data[r"patients"] = len(instance.patients)
    data[r"total treatment \mbox{appointments}"] = sum(
        sum(p.treatments.values()) for p in instance.patients.values()
    )
    data["planning horizon (in days)"] = str(instance.horizon_length)
    data["daily resolution (in min)"] = str(int(instance.time_slot_length.hours * 60))
    data[r"resources"] = len(instance.resources)
    data[r"treatments"] = len(instance.treatments)
    avg_treat = 0
    avg_treat_num_days = 0
    for p in instance.patients.values():
        avg_treat += sum(p.treatments.values()) / p.length_of_stay
        avg_treat_num_days += 1

    data[r"avg. num treatments per patient per day"] = round(
        avg_treat / avg_treat_num_days, 2
    )

    return data


frame = pd.DataFrame(table_data)
seen_instances = set()
instance_list = []

replace_words = {
    "instance_": "",
    "_": r"\_",
    r".txt": "",
    r"\_less\_prescriptions": r"\_a",
    r"\_more\_resources": r"\_b",
    r"\_short\_timeframe": r"\_c",
}
average_num_treatments = 0
average_num_n = 0
instance_path_folder = "data/comp_study_004/"
for instance_file in os.listdir(instance_path_folder):
    if "t15" not in instance_file:
        continue
    if "low" not in instance_file:
        continue
    instance_path = instance_path_folder + instance_file
    instance = create_instance_from_file(instance_path)
    instance_data = calculate_data(instance)
    instance_file_name = copy(instance_file)
    for word in replace_words:
        instance_file_name = instance_file_name.replace(word, replace_words[word])

    if instance_file_name in seen_instances:
        continue
    instance_data["instance"] = instance_file_name
    seen_instances.add(instance_file_name)
    instance_list.append(instance_data)


headers = [
    r"instance",
    r"patients",
    r"total treatment \mbox{appointments}",
    r"avg. num treatments per patient per day",
    r"resources",
]

latex = """
\\setlength{\\tabcolsep}{12pt}
\\begin{table}[t!]
\\begin{tabularx}{\\textwidth}{%COLUMNS}
\\hline
%HEADER
\\hline
%DATA
\\end{tabularx}
\\caption{Key instance characteristics of the base instances used in the computational study with low treatments assigned}
\\label{tab:instance_characteristics}
\\end{table}
"""
latex = latex.replace("%COLUMNS", "p{2.2cm}p{1.5cm}XXp{1.5cm}")


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
