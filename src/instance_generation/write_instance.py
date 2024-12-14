from typing import List, Tuple, Set, Dict, Optional
from src.time import Duration, DayHour
from src.resource import Resource, ResourceGroup
from src.treatments import Treatment
from src.patients import Patient
import math
import os
from src.instance import Instance
import os
from typing import Any
from collections import defaultdict


def write_instance_to_file(file_path: str, instance: Instance):
    # Extract the directory path from the file path
    directory = os.path.dirname(file_path)

    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Open the file for writing
    with open(file_path, "w") as file:
        # Write the [INSTANCE] section
        file.write("[INSTANCE]\n")
        file.write(f"num_beds: {instance.beds_capacity}\n")
        file.write(f"workday_start: {instance.workday_start.hour}\n")
        file.write(f"workday_end: {instance.workday_end.hour}\n")
        file.write(f"day_start: 0\n")
        file.write(f"rolling_window_length: {instance.rolling_window_length}\n")
        rolling_window_days_str = ", ".join(map(str, instance.rolling_window_days))
        file.write(f"rolling_window_days: [{rolling_window_days_str}]\n\n")

        # Write conflict groups
        file.write("conflict_groups: [")
        for group in instance.conflict_groups:
            tids = ", ".join(str(t.id) for t in group)
            file.write(f"{{{tids}}},")
        file.write("]\n")

        # Write [DATA: RESOURCE_GROUPS] section
        file.write("[DATA: RESOURCE_GROUPS]: rgid, name\n")
        for rgid, rg in instance.resource_groups.items():
            file.write(f"{rgid}; {rg.name}\n")
        file.write("\n")

        # Write [DATA: RESOURCES] section
        file.write("[DATA: RESOURCES]: rid, rgid, name, unavailable_time_slots\n")
        for rid, resource in instance.resources.items():
            rgid = resource.resource_group.id
            unavailable_slots_str = (
                f"{resource.unavailable_time_slots}"
                if resource.unavailable_time_slots
                else "None"
            )
            file.write(f"{rid}; {rgid}; {resource.name}; {unavailable_slots_str}\n")
        file.write("\n")

        # Write [DATA: TREATMENTS] section
        file.write(
            "[DATA: TREATMENTS]: tid, num_participants, name, duration, resources\n"
        )
        for tid, treatment in instance.treatments.items():
            duration = f"Duration(hours={treatment.duration.hours})"
            resources_str = (
                "{"
                + ", ".join(
                    f"{rg.id}: ({treatment.resources[rg]}, {treatment.loyalty[rg]})"
                    for rg in treatment.resources
                )
                + "}"
            )
            file.write(
                f"{tid}; {treatment.max_num_participants}; {treatment.name}; {duration}; {resources_str}\n"
            )
        file.write("\n")

        # Write [DATA: PATIENTS] section
        file.write(
            "[DATA: PATIENTS]: pid, name, treatments, length_of_stay, earliest_admission_date, admitted_before_date, already_admitted, already_resource_loyal, already_scheduled_treatments\n"
        )
        for pid, patient in instance.patients.items():
            treatments_str = (
                "{"
                + ", ".join(f"{t.id}: {n}" for t, n in patient.treatments.items())
                + "}"
            )
            earliest_admission = f"DayHour(day={patient.earliest_admission_date.day}, hour={patient.earliest_admission_date.hour})"
            admitted_before = f"DayHour(day={patient.admitted_before_date.day}, hour={patient.admitted_before_date.hour})"
            resource_loyal_str = (
                "{"
                + ", ".join(
                    f"({t.id}, {rg.id}): {[r.id for r in resources]}"
                    for (t, rg), resources in patient.already_resource_loyal.items()
                )
                + "}"
            )
            scheduled_treatments_str = (
                "["
                + ", ".join(
                    f"({t.id}, {str(n)})"
                    for t, n in patient.already_scheduled_treatments.items()
                )
                + "]"
            )
            file.write(
                f"{pid}; {patient.name}; {treatments_str}; {patient.length_of_stay}; {earliest_admission}; {admitted_before}; {patient.already_admitted}; {resource_loyal_str}; {scheduled_treatments_str}\n"
            )
