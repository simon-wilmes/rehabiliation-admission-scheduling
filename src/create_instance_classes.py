import math
import random
from typing import Tuple
from typing import List, Tuple, Set, Dict, Optional
from src.time import Duration, DayHour
from src.resource import Resource, ResourceGroup
from src.treatments import Treatment
from src.patients import Patient
from src.types import RID, RGID, TID, PID
from src.write_instance import generate_instance_file


def random_day_hour(day_range: Tuple[int, int], hour_range: Tuple[int, int]) -> DayHour:
    day = random.randint(*day_range)
    hour = random.uniform(*hour_range)
    return DayHour(day, hour)


def random_duration(
    hours_range: Tuple[int, int], minutes_range: Tuple[int, int] = (0, 59)
) -> Duration:
    # Ensure the input ranges are tuples
    if isinstance(hours_range, int):
        hours_range = (hours_range, hours_range)
    if isinstance(minutes_range, int):
        minutes_range = (minutes_range, minutes_range)
    # Generate random values within the provided ranges
    hours = random.randint(*hours_range)
    minutes = random.randint(*minutes_range)
    return Duration(hours, minutes / 60)


def random_resource_group(rgid: RGID, name: str) -> ResourceGroup:
    return ResourceGroup(rgid, name)


def random_resource(
    rid: RID,
    resource_group: ResourceGroup,
    name: str,
    time_slots: List[Tuple[DayHour, Duration]] = None,
) -> Resource:
    if time_slots is None:
        time_slots = []
    return Resource(rid, resource_group, name, unavailable_time_slots=time_slots)


def random_treatment(
    tid: TID,
    name: str,
    duration: Duration,
    num_participants: int,
    resources: Dict[ResourceGroup, Tuple[int, bool]],
) -> Treatment:
    return Treatment(tid, num_participants, duration, name, resources)


def random_patient(
    pid: PID,
    name: str,
    treatments: Dict[Treatment, int],
    length_of_stay: int,
    earliest_admission_date: DayHour,
    admitted_before_date: DayHour,
    already_admitted: bool,
) -> Patient:
    return Patient(
        pid,
        treatments,
        length_of_stay,
        earliest_admission_date,
        admitted_before_date,
        already_admitted,
        name=name,
    )


def generate_random_instance(
    file_path: str,
    num_beds_range: Tuple[int, int],
    workday_start_range: Tuple[int, int],
    workday_end_range: Tuple[int, int],
    day_start: int,
    time_interval_length: int,
    rolling_window_length_range: Tuple[int, int],
    rolling_windows_days: List[Tuple[int, int]],
    conflict_group_number_range: Tuple[int, int],
    conflict_group_treatments_range: Tuple[int, int],
    resource_group_names: List[str],
    resource_names: List[str],
    treatment_names: List[str],
    patient_names: List[str],
    num_resources: int,
    num_treatments: int,
    num_patients: int,
    already_admitted_chance: float,
    resource_time_slot_range: Tuple[int, int],
    treatment_duration_range: Tuple[int, int],
    treatment_capacity_range: Tuple[int, int],
    patient_length_of_stay_range: Tuple[int, int],
    admission_day_range: Tuple[int, int],
    admission_hour_range: Tuple[int, int],
    needed_treatments_range: Tuple[int, int],
    treatment_repetition_range: Tuple[int, int],
):
    # Generate general settings
    num_beds = random.randint(*num_beds_range)
    workday_start = random.randint(*workday_start_range)
    workday_end = random.randint(*workday_end_range)
    rolling_window_length = random.randint(*rolling_window_length_range)
    time_intervals = time_interval_length / 60

    # Generate Conflict Groups
    conflict_groups = []
    for i in range(random.randint(*conflict_group_number_range)):
        conflict_groups.append(
            random.sample(
                range(num_treatments), random.randint(*conflict_group_treatments_range)
            )
        )
    # Generate Resource Groups
    resource_groups = [
        random_resource_group(i, name) for i, name in enumerate(resource_group_names)
    ]

    # Generate Resources, ensuring each name in resource_names is used exactly once
    resources = []
    resource_name_cycle = (resource_names * (num_resources // len(resource_names) + 1))[
        :num_resources
    ]
    for i, name in enumerate(resource_name_cycle):
        resource_group = random.choice(resource_groups)
        time_slots = [
            (
                random_day_hour(admission_day_range, admission_hour_range),
                random_duration(*treatment_duration_range),
            )
            for _ in range(random.randint(*resource_time_slot_range))
        ]
        resources.append(
            random_resource(
                rid=i, resource_group=resource_group, name=name, time_slots=time_slots
            )
        )

    # Generate Treatments, ensuring each name in treatment_names is used exactly once
    treatments = []
    treatment_name_cycle = (
        treatment_names * (num_treatments // len(treatment_names) + 1)
    )[:num_treatments]
    for i, name in enumerate(treatment_name_cycle):
        duration = random_duration(*treatment_duration_range)
        num_participants = random.randint(
            treatment_capacity_range[0], treatment_capacity_range[1]
        )
        resources_required = {
            random.choice(resource_groups): (
                random.randint(1, len(resource_group_names)),
                bool(random.getrandbits(1)),
            )
        }
        treatments.append(
            random_treatment(
                tid=i,
                name=name,
                duration=duration,
                num_participants=num_participants,
                resources=resources_required,
            )
        )

    # Generate Patients, ensuring each name in patient_names is used exactly once
    patients = []
    patient_name_cycle = (patient_names * (num_patients // len(patient_names) + 1))[
        :num_patients
    ]
    for i, name in enumerate(patient_name_cycle):
        treatment_list_temp = random.sample(
            treatments,
            random.randint(needed_treatments_range[0], needed_treatments_range[1]),
        )
        treatments_needed = {
            treatment_list_temp[j]: random.randint(
                treatment_repetition_range[0], treatment_repetition_range[1]
            )
            for j in range(len(treatment_list_temp))
        }
        length_of_stay = random.randint(*patient_length_of_stay_range)
        v1, v2 = random.sample(
            range(list(admission_day_range)[0], list(admission_day_range)[1]), 2
        )
        earliest_admission_date_temp, admitted_before_date_temp = sorted([v1, v2])
        earliest_admission_date = random_day_hour(
            (earliest_admission_date_temp, earliest_admission_date_temp),
            admission_hour_range,
        )
        admitted_before_date = random_day_hour(
            (admitted_before_date_temp, admitted_before_date_temp), admission_hour_range
        )
        already_admitted = random.random() <= already_admitted_chance
        if already_admitted == True:
            earliest_admission_date = DayHour(day_start, workday_start, 0)
            admitted_before_date = DayHour(day_start, workday_start, 0)
            # already_admitted = random_day_hour(
            #    (-length_of_stay, 0), (workday_start, workday_end)
            # )
        patients.append(
            random_patient(
                pid=i,
                name=name,
                treatments=treatments_needed,
                length_of_stay=length_of_stay,
                earliest_admission_date=earliest_admission_date,
                admitted_before_date=admitted_before_date,
                already_admitted=already_admitted,
            )
        )

    # Generate instance file
    generate_instance_file(
        file_path=file_path,
        num_beds=num_beds,
        workday_start=workday_start,
        workday_end=workday_end,
        day_start=day_start,
        time_intervals=time_intervals,
        rolling_window_length=rolling_window_length,
        rolling_windows_days=rolling_windows_days,
        conflict_groups=conflict_groups,
        resource_groups=resource_groups,
        resources=resources,
        treatments=treatments,
        patients=patients,
    )


instance_file_number = 3
number_of_instances = 20
num_patients = 20
num_treatments = 5
num_resources = 5
for i in range(number_of_instances):
    generate_random_instance(
        file_path=f"data/inst{instance_file_number:03}/instance_data{i+1}.txt",
        num_beds_range=(15, 15),
        workday_start_range=(8, 8),
        workday_end_range=(18, 18),
        day_start=0,
        time_interval_length=2,
        rolling_window_length_range=(8, 8),
        rolling_windows_days=[(0, 4)],
        conflict_group_number_range=[0, 2],
        conflict_group_treatments_range=[2, 3],
        resource_group_names=["RG_1", "RG_2", "RG_3"],
        resource_names=[f"R_{i}" for i in range(0, num_resources)],
        treatment_names=[f"T_{i}" for i in range(0, num_treatments)],
        patient_names=[f"P_{i}" for i in range(0, num_patients)],
        num_resources=num_resources,
        num_treatments=num_treatments,
        num_patients=num_patients,
        already_admitted_chance=1,
        resource_time_slot_range=(1, 5),
        treatment_duration_range=((1, 3), (0, 59)),
        treatment_capacity_range=(1, 4),
        patient_length_of_stay_range=(3, 10),
        admission_day_range=(0, 10),
        admission_hour_range=(8, 17),
        needed_treatments_range=(2, 5),
        treatment_repetition_range=(1, 5),
    )

'''
def save_code_to_file(filepath: str):
    code_text = """\
instance_file_number = 2
number_of_instances = 20
num_patients = 10
num_treatments = 5
num_resources = 5
for i in range(number_of_instances):
    generate_random_instance(
        file_path=f"data/inst{instance_file_number:03}/instance_data{i+1}.txt",
        num_beds_range=(10, 10),
        workday_start_range=(8, 8),
        workday_end_range=(18, 18),
        day_start=0,
        time_interval_length=15,
        rolling_window_length_range=(8, 8),
        rolling_windows_days=[(0, 4)],
        conflict_group_number_range=[0, 1],
        conflict_group_treatments_range=[2, 3],
        resource_group_names=["RG_1", "RG_2", "RG_3"],
        resource_names=[f"R_{i}" for i in range(0, num_resources)],
        treatment_names=[f"T_{i}" for i in range(0, num_treatments)],
        patient_names=[f"P_{i}" for i in range(0, num_patients)],
        num_resources=num_resources,
        num_treatments=num_treatments,
        num_patients=num_patients,
        already_admitted_chance=1,
        resource_time_slot_range=(1, 5),
        treatment_duration_range=((1, 3), (0, 59)),
        treatment_capacity_range=(1, 4),
        patient_length_of_stay_range=(3, 10),
        admission_day_range=(0, 10),
        admission_hour_range=(8, 17),
        needed_treatments_range=(2, 5),
        treatment_repetition_range=(1, 5),
    )
"""

    # Write the code_text to a file
    with open("data/instance details/inst{instance_file_number:03}", "w") as file:
        file.write(code_text)
        # further details:
        file.write(" ")


save_code_to_file("parameters_reference.txt")
'''
