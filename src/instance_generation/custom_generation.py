import random
import math
from typing import Any, Dict
from collections import defaultdict
from src.instance import Instance
from src.resource import Resource, ResourceGroup
from src.treatments import Treatment
from src.patients import Patient
from src.time import DayHour, Duration
import os
from src.instance_generation.write_instance import write_instance_to_file


def generate_unavailability_slots(
    total_unavailability_time: float,
    duration_range: tuple[int, int],
    time_granularity_minutes: int,
    workday_start: float,
    workday_end: float,
    total_days: int,
) -> list[tuple[DayHour, DayHour, None]]:
    """
    Generate unavailability time slots for a resource.

    :param total_unavailability_time: Total unavailability time in hours.
    :param duration_range: Tuple specifying the min and max duration in minutes.
    :param time_granularity_minutes: Time granularity in minutes.
    :param workday_start: Workday start time in hours.
    :param workday_end: Workday end time in hours.
    :param total_days: Total number of days.
    :return: List of unavailability time slots.
    """
    random.seed(0)
    unavailability_slots = []
    remaining_unavailability_time = total_unavailability_time * 60  # Convert to minutes
    min_duration, max_duration = duration_range

    # Align durations with time granularity
    min_duration = max(
        time_granularity_minutes,
        int(min_duration / time_granularity_minutes) * time_granularity_minutes,
    )
    max_duration = max_duration - (max_duration % time_granularity_minutes)

    # Distribute unavailability across days
    for day in range(total_days):
        daily_unavailability = 0
        max_daily_unavailability = (
            workday_end - workday_start
        ) * 60  # Total minutes in a workday
        while (
            remaining_unavailability_time > 0
            and daily_unavailability < max_daily_unavailability
        ):
            # Randomly select a duration
            duration = random.randint(min_duration, max_duration)
            duration = int(
                round(duration / time_granularity_minutes) * time_granularity_minutes
            )
            duration = min(duration, remaining_unavailability_time)
            duration = min(
                duration, max_daily_unavailability - daily_unavailability
            )  # Cannot exceed workday duration

            # Randomly select a start time within workday
            available_minutes = int(max_daily_unavailability - duration)
            if available_minutes <= 0:
                break
            start_minutes = random.randint(0, available_minutes)
            start_minutes = int(
                round(start_minutes / time_granularity_minutes)
                * time_granularity_minutes
            )

            # Create DayHour objects
            start_hour = int(workday_start + start_minutes / 60)
            start_minute = int(start_minutes % 60)
            end_hour = int(workday_start + (start_minutes + duration) / 60)
            end_minute = int((start_minutes + duration) % 60)

            start_time = DayHour(day=day, hour=start_hour, minutes=start_minute)
            end_time = DayHour(day=day, hour=end_hour, minutes=end_minute)

            # Append the unavailability slot
            unavailability_slots.append((start_time, end_time, None))

            remaining_unavailability_time -= duration
            daily_unavailability += duration

            if remaining_unavailability_time <= 0:
                break

    return unavailability_slots


def generate_instance(params: dict[str, Any]) -> Instance:
    # Extract parameters
    num_patients = params.get("num_patients", 100)
    num_resource_groups = params.get("num_resource_groups", 3)
    num_treatments = params.get("num_treatments", 10)
    workday_start = params.get("workday_start", 8)  # Hour
    workday_end = params.get("workday_end", 18)  # Hour
    length_of_stay_range = params.get("length_of_stay_range", (3, 10))
    # Removed treatments_per_patient_range
    treatment_duration_distribution = params.get(
        "treatment_duration_distribution", {"mean": 1, "std_dev": 0.25}
    )  # Hours
    resource_usage_target = params.get("resource_usage_target", 0.7)
    admitted_patient_percentage = params.get("admitted_patient_percentage", 0.3)
    resource_group_loyalty_probability_range = params.get(
        "resource_group_loyalty_probability_range", (0.1, 0.9)
    )
    time_granularity_minutes = params.get("time_granularity_minutes", 15)
    num_beds = params.get("num_beds", 50)  # Fixed number of beds less than num_patients
    max_admission_day = params.get("max_admission_day", 30)  # Planning horizon in days
    resource_group_unavailability_percentages = params.get(
        "resource_group_unavailability_percentages",
        {},  # Default to empty dict; will assign default values later
    )
    num_resources_per_treatment = params.get("num_resources_per_treatment", (1, 3))
    unavailability_duration_range = params.get(
        "unavailability_duration_range", (30, 120)
    )  # In minutes
    average_treatments_per_day = params.get("average_treatments_per_day", 3)

    # Ensure workday times are aligned with time granularity
    def align_time(value):
        return (
            int(value * 60 / time_granularity_minutes) * time_granularity_minutes / 60
        )

    workday_start = align_time(workday_start)
    workday_end = align_time(workday_end)
    workday_duration_hours = workday_end - workday_start

    # Initialize ID counters
    next_rgid = 0
    next_rid = 0
    next_tid = 0
    next_pid = 0

    # Create Resource Groups with sampled loyalty probabilities
    resource_groups = {}
    for _ in range(num_resource_groups):
        rgid = next_rgid
        name = f"RG_{rgid}"
        # Sample loyalty probability from the specified range
        min_prob, max_prob = resource_group_loyalty_probability_range
        resource_loyalty_probability = random.uniform(min_prob, max_prob)
        rg = ResourceGroup(rgid=rgid, name=name)
        rg.resource_loyalty_probability = resource_loyalty_probability  # type: ignore

        # Assign unavailability percentage to the resource group
        # If not specified, default to 0% unavailability
        rg_unavailability_percentage = resource_group_unavailability_percentages.get(
            rgid, 0.0
        )
        rg.unavailability_percentage = rg_unavailability_percentage  # type: ignore

        resource_groups[rgid] = rg
        next_rgid += 1

    # Generate Treatments
    treatments = {}
    treatments_list = []

    def generate_duration():
        # Generate duration in minutes aligned with granularity
        mean_duration = (
            treatment_duration_distribution["mean"] * 60
        )  # Convert to minutes
        std_dev_duration = (
            treatment_duration_distribution["std_dev"] * 60
        )  # Convert to minutes
        duration_minutes = random.gauss(mean_duration, std_dev_duration)
        duration_minutes = max(time_granularity_minutes, duration_minutes)
        duration_minutes = int(
            round(duration_minutes / time_granularity_minutes)
            * time_granularity_minutes
        )
        hours = int(duration_minutes // 60)
        minutes = int(duration_minutes % 60)
        return Duration(hours=hours, minutes=minutes)

    for _ in range(num_treatments):
        tid = next_tid
        num_participants = random.randint(1, 10)
        duration = generate_duration()
        name = f"Treatment_{tid}"
        resources_required = {}
        loyalty = {}
        for rg in resource_groups.values():
            if random.random() < 0.5:
                num_resources_needed = random.randint(*num_resources_per_treatment)
                # Determine if the resource is loyal based on the group's loyalty probability
                requires_loyalty = random.random() < rg.resource_loyalty_probability
                resources_required[rg] = (num_resources_needed, requires_loyalty)
        treatment = Treatment(
            tid=tid,
            num_participants=num_participants,
            duration=duration,
            name=name,
            resources=resources_required,
        )
        treatments[tid] = treatment
        treatments_list.append(treatment)
        next_tid += 1

    # Prepare data for Instance
    instance_data = {
        "num_beds": num_beds,
        "workday_start": DayHour(day=0, hour=int(workday_start)),
        "workday_end": DayHour(day=0, hour=int(workday_end)),
        "rolling_window_length": 7,  # Example value
        "rolling_windows_days": [0, 5, 10, 15, 20],  # Example value
        "conflict_groups": [],  # Could be generated if needed
    }

    # Create dictionaries for Instance constructor
    resource_groups_dict = {rg.id: rg for rg in resource_groups.values()}
    treatments_dict = {t.id: t for t in treatments_list}

    # Create Instance
    instance = Instance(
        instance_data=instance_data,
        resource_groups=resource_groups_dict,
        treatments=treatments_dict,
        resources={},
        patients={},
    )

    return instance

