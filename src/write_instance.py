from typing import List, Tuple, Set, Dict, Optional
from src.time import Duration, DayHour
from src.resource import Resource, ResourceGroup
from src.treatments import Treatment
from src.patients import Patient
import math


def round_up_fraction(value, fraction):
    return math.ceil(value / fraction) * fraction


def round_down_fraction(value, fraction):
    return math.floor(value / fraction) * fraction


def generate_instance_file(
    file_path: str,
    num_beds: int,
    workday_start: int,
    workday_end: int,
    day_start: int,
    time_intervals: int,
    rolling_window_length: int,
    rolling_windows_days: List[Tuple[int, int]],
    conflict_groups: List[Set[int]],
    resource_groups: List,
    resources: List,
    treatments: List,
    patients: List,
):
    with open(file_path, "w") as file:
        # Writing general instance settings
        file.write("[INSTANCE]\n")
        file.write(f"num_beds: {num_beds}\n")
        file.write(f"workday_start: {workday_start} # uhr\n")
        file.write(f"workday_end: {workday_end} # uhr\n")
        file.write(f"day_start: {day_start}\n")
        file.write(f"rolling_window_length: {rolling_window_length}\n")
        file.write(f"rolling_windows_days: {rolling_windows_days}\n")
        file.write(
            "# rolling_windows_days is a list of tuples where the first entry says the \n"
        )
        file.write(
            "# start day and the second entry says the interval of repetition of None if only once\n"
        )
        file.write(
            "# conflict groups is a list of sets where each set represents a single conflict group\n"
        )
        file.write(f"conflict_groups: {conflict_groups}\n\n")

        # Writing Resource Groups data
        file.write("[DATA: RESOURCE_GROUPS]: rgid, name\n")
        file.write("# 'rgid' is of type int (must be unique)\n")
        file.write("# 'name' is of type str\n")
        for rg in resource_groups:
            file.write(f"{rg.id}; {rg.name}\n")
        file.write("\n")

        # Writing Resources data
        file.write("[DATA: RESOURCES]: rid, rgid, name, unavailable_time_slots\n")
        file.write("# 'rid' is of type int (must be unique)\n")
        file.write("# 'rgid' is of type int: refers to the rgid of a resource_group\n")
        file.write("# 'name' is of type str\n")
        file.write(
            "# 'unavailable_time_slots' is of type: list[tuple[DayHour, Duration, int | None]]\n"
        )
        file.write("# where the syntax is the first entry says the start date,\n")
        file.write("# the second entry is the end date of the unavailability\n")
        file.write(
            "# and the last entry is the interval of days of repetition, none if only once\n"
        )
        for resource in resources:
            file.write(
                f"{resource.id}; {resource.resource_group.id}; {resource.name}; ["
            )
            slots = [
                f"(DayHour(day={slot[0].to_tuple()[0]}, hour={round_up_fraction(slot[0].to_tuple()[1],time_intervals)}), Duration(hours={round_down_fraction(slot[1].hours,time_intervals)}))"
                for slot in resource.unavailable_time_slots
            ]
            file.write(", ".join(slots) + "]\n")
        file.write("\n")

        # Writing Treatments data
        file.write(
            "[DATA: TREATMENTS]: tid, num_participants, name, duration, resources\n"
        )
        file.write("# 'tid' is of type int (must be unique)\n")
        file.write("# 'num_participants' is of type int\n")
        file.write("# 'name' is of type str\n")
        file.write("# 'duration' is of type Duration\n")
        file.write(
            "# 'resources' is of type dict[RGID, tuple[number_of_resources, requires_loyalty]]\n"
        )
        for treatment in treatments:
            file.write(
                f"{treatment.id}; {treatment.num_participants}; {treatment.name}; Duration(hours={round_down_fraction(treatment.duration.hours,time_intervals)}); {treatment.resources}\n"
            )
        file.write("\n")

        # Writing Patients data
        file.write(
            "[DATA: PATIENTS]: pid, name, treatments, length_of_stay, earliest_admission_date, "
            "admitted_before_date, already_admitted_date, already_resource_loyal, already_scheduled_treatments\n"
        )
        file.write("# 'pid' is of type int (must be unique)\n")
        file.write("# 'name' is of type string (patient's name)\n")
        file.write(
            "# 'treatments' is a dictionary where keys are treatment IDs (int) and values are number of treatments required\n"
        )
        file.write("# 'length_of_stay' is of type int (total length of stay in days)\n")
        file.write(
            "# 'earliest_admission_date' is of type DayHour (earliest possible admission date)\n"
        )
        file.write(
            "# 'admitted_before_date' is of type DayHour (latest possible admission date)\n"
        )
        file.write(
            "# 'already_admitted' is of type DayHour and refers to the day when the patient was admitted is None if not admitted (most of the time <0, as in the past admitted)\n"
        )
        file.write(
            "# 'already_resource_loyal' is a dictionary where the keys are (tuples) of treatment TID and RGID and values is a the list of RID that are loyal, this means that the list must have length of required resources of RGID for treatment TID\n"
        )
        file.write(
            "# 'already_scheduled_treatments' is a list of tuples where each tuple contains a treatment ID (int) and the number of times it was already scheduled in the past\n"
        )
        for patient in patients:
            file.write(
                f"{patient.id}; {patient.name}; {patient.treatments}; {patient.length_of_stay}; "
                f"DayHour(day={patient.earliest_admission_date.to_tuple()[0]},hour={round_down_fraction(patient.earliest_admission_date.to_tuple()[1],time_intervals)}); DayHour(day={patient.admitted_before_date.to_tuple()[0]},hour={round_down_fraction(patient.admitted_before_date.to_tuple()[1],time_intervals)}); {patient.already_admitted}; "
                f"{patient.already_resource_loyal}\n"
            )


# Example of usage
# Assuming you have already created objects for resource_groups, resources, treatments, and patients:

"""
generate_instance_file(
    file_path="data/inst001/instance_data.txt",
    num_beds=2,
    workday_start=8,
    workday_end=17,
    day_start=0,
    rolling_window_length=7,
    rolling_windows_days=[(0, 4)],
    conflict_groups=[{0}],
    resource_groups=resource_groups,  # List of ResourceGroup instances
    resources=resources,  # List of Resource instances
    treatments=treatments,  # List of Treatment instances
    patients=patients,  # List of Patient instances
)
"""
