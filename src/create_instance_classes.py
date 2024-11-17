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
    earliest_admission: DayHour,
    latest_admission: DayHour,
    already_admitted: bool,
) -> Patient:
    return Patient(
        pid,
        treatments,
        length_of_stay,
        earliest_admission,
        latest_admission,
        already_admitted,
        name=name,
    )


def generate_random_instance(
    file_path: str,
    num_beds_range: Tuple[int, int],
    workday_start_range: Tuple[int, int],
    workday_end_range: Tuple[int, int],
    day_start: int,
    rolling_window_length_range: Tuple[int, int],
    rolling_windows_days: List[Tuple[int, int]],
    conflict_groups: List[Set[int]],
    resource_group_names: List[str],
    resource_names: List[str],
    treatment_names: List[str],
    patient_names: List[str],
    num_resources: int,
    num_treatments: int,
    num_patients: int,
    resource_time_slot_range: Tuple[int, int],
    treatment_duration_range: Tuple[int, int],
    patient_length_of_stay_range: Tuple[int, int],
    admission_day_range: Tuple[int, int],
    admission_hour_range: Tuple[int, int],
):
    # Generate general settings
    num_beds = random.randint(*num_beds_range)
    workday_start = random.randint(*workday_start_range)
    workday_end = random.randint(*workday_end_range)
    rolling_window_length = random.randint(*rolling_window_length_range)

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
        num_participants = random.randint(1, 5)
        resources_required = {
            random.choice(resource_groups): (
                random.randint(1, 3),
                bool(random.getrandbits(1)),
            )
            for _ in range(random.randint(1, 3))
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
        treatments_needed = {
            random.choice(treatments): random.randint(1, 3)
            for _ in range(random.randint(1, 3))
        }
        length_of_stay = random.randint(*patient_length_of_stay_range)
        earliest_admission = random_day_hour(admission_day_range, admission_hour_range)
        latest_admission = random_day_hour(admission_day_range, admission_hour_range)
        already_admitted = bool(random.getrandbits(1))
        patients.append(
            random_patient(
                pid=i,
                name=name,
                treatments=treatments_needed,
                length_of_stay=length_of_stay,
                earliest_admission=earliest_admission,
                latest_admission=latest_admission,
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
        rolling_window_length=rolling_window_length,
        rolling_windows_days=rolling_windows_days,
        conflict_groups=conflict_groups,
        resource_groups=resource_groups,
        resources=resources,
        treatments=treatments,
        patients=patients,
    )


generate_random_instance(
    file_path="data/inst001/instance_data.txt",
    num_beds_range=(1, 5),
    workday_start_range=(8, 10),
    workday_end_range=(16, 18),
    day_start=0,
    rolling_window_length_range=(5, 10),
    rolling_windows_days=[(0, 4)],
    conflict_groups=[{0}],
    resource_group_names=["Radiology", "ICU", "Lab"],
    resource_names=["X-Ray", "CT Scanner", "Ultrasound", "Ventilator", "MRI"],
    treatment_names=["Therapy A", "Therapy B", "Therapy C"],
    patient_names=["Alice", "Bob", "Charlie", "Diana"],
    num_resources=10,
    num_treatments=5,
    num_patients=20,
    resource_time_slot_range=(1, 3),
    treatment_duration_range=(1, 5),
    patient_length_of_stay_range=(3, 10),
    admission_day_range=(0, 10),
    admission_hour_range=(8, 17),
)
