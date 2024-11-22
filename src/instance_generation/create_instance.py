import random
import math
from typing import Any, Dict
from collections import defaultdict
from src.instance import Instance
from src.resource import Resource, ResourceGroup
from src.treatments import Treatment
from src.patients import Patient
from src.time import DayHour, Duration


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
    length_of_stay_range = params.get("length_of_stay_range", (3, 7))
    treatments_per_patient_range = params.get("treatments_per_patient_range", (1, 5))
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

    # Now generate Patients
    total_patients = num_patients
    num_already_admitted = int(total_patients * admitted_patient_percentage)
    num_new_patients = total_patients - num_already_admitted

    patients_dict = {}
    total_resource_demand = defaultdict(float)
    total_treatment_demand = defaultdict(int)
    length_of_stays = []
    # Collect length of stay data for admission window calculation
    for _ in range(total_patients):
        length_of_stay = random.randint(*length_of_stay_range)
        length_of_stays.append(length_of_stay)

    # Calculate total bed days needed
    total_bed_days_needed = sum(length_of_stays)

    # Calculate total bed days available
    planning_horizon_days = max_admission_day
    total_bed_days_available = num_beds * planning_horizon_days

    # Check feasibility
    if total_bed_days_needed > total_bed_days_available:
        # Adjust patients' length of stay or number of patients
        scaling_factor = total_bed_days_available / total_bed_days_needed
        adjusted_length_of_stays = [
            max(1, int(ls * scaling_factor)) for ls in length_of_stays
        ]
    else:
        adjusted_length_of_stays = length_of_stays

    # Distribute admissions over the planning horizon
    bed_occupancy = [0] * (planning_horizon_days + max(adjusted_length_of_stays))
    patient_admission_days = []

    for idx, length_of_stay in enumerate(adjusted_length_of_stays):
        # Find earliest day where patient can be admitted without exceeding bed capacity
        for admission_day in range(planning_horizon_days):
            can_admit = True
            for day in range(admission_day, admission_day + length_of_stay):
                if bed_occupancy[day] >= num_beds:
                    can_admit = False
                    break
            if can_admit:
                # Admit patient on this day
                for day in range(admission_day, admission_day + length_of_stay):
                    bed_occupancy[day] += 1
                patient_admission_days.append(admission_day)
                break
        else:
            # Could not find an admission day, extend planning horizon
            admission_day = len(bed_occupancy) - length_of_stay
            for day in range(admission_day, admission_day + length_of_stay):
                if day >= len(bed_occupancy):
                    bed_occupancy.append(1)
                else:
                    bed_occupancy[day] += 1
            patient_admission_days.append(admission_day)

    # Generate patients with assigned admission days
    for idx in range(total_patients):
        pid = next_pid
        already_admitted = idx < num_already_admitted
        length_of_stay = adjusted_length_of_stays[idx]
        admission_day = patient_admission_days[idx]

        earliest_admission_day = admission_day
        admitted_before_day = earliest_admission_day + random.randint(
            1, 5
        )  # Small window

        earliest_admission_date = DayHour(day=earliest_admission_day, hour=0, minutes=0)
        admitted_before_date = DayHour(day=admitted_before_day, hour=0, minutes=0)
        name = f"Patient_{pid}"

        treatments_required = {}
        num_treatments_assigned = random.randint(*treatments_per_patient_range)
        selected_treatments = random.sample(treatments_list, num_treatments_assigned)
        already_scheduled_treatments = []
        treatments_with_loyalty = set()

        for treatment in selected_treatments:
            num_sessions = random.randint(1, 5)
            if already_admitted:
                sessions_completed = random.randint(1, num_sessions)
                num_sessions_remaining = num_sessions - sessions_completed
                already_scheduled_treatments.append((treatment, sessions_completed))
            else:
                sessions_completed = 0
                num_sessions_remaining = num_sessions

            num_sessions_remaining = max(0, num_sessions_remaining)

            if num_sessions_remaining >= 0:
                treatments_required[treatment] = num_sessions

            # Update total resource demand based on remaining sessions
            duration_hours = treatment.duration.hours
            for rg, (num_resources_needed) in treatment.resources.items():
                demand = num_sessions_remaining * duration_hours * num_resources_needed
                total_resource_demand[rg.id] += demand
            total_treatment_demand[treatment.id] += num_sessions_remaining

            # Store treatments that require resource loyalty for later assignment
            if already_admitted and sessions_completed > 0:
                for rg in treatment.resources.keys():
                    if treatment.loyalty[rg]:
                        treatments_with_loyalty.add((treatment, rg))

        # Create Patient object
        patient = Patient(
            pid=pid,
            treatments=treatments_required,
            length_of_stay=length_of_stay,
            earliest_admission_date=earliest_admission_date,
            admitted_before_date=admitted_before_date,
            already_admitted=already_admitted,
            already_resource_loyal={},  # Will be updated later
            already_scheduled_treatments=already_scheduled_treatments,  # Now a list of tuples
            name=name,
        )
        # Store treatments with loyalty for later processing
        patient.treatments_with_loyalty = treatments_with_loyalty  # type: ignore
        patients_dict[pid] = patient
        next_pid += 1

    # Calculate total work time per resource
    work_hours_per_day = workday_duration_hours
    total_days = planning_horizon_days
    total_work_time_per_resource = work_hours_per_day * total_days

    # Determine number of resources needed per resource group
    resources_needed = {}
    for rgid, total_demand in total_resource_demand.items():
        rg = resource_groups[rgid]
        # Use the group's unavailability percentage to estimate average available hours per resource
        unavailability_percentage = rg.unavailability_percentage
        adjusted_available_hours_per_unit = total_work_time_per_resource * (
            1 - unavailability_percentage
        )
        target_hours_per_unit = (
            adjusted_available_hours_per_unit * resource_usage_target
        )
        num_resources = max(1, math.ceil(total_demand / target_hours_per_unit))
        # num_resources is bound below by the minimum number needed by any treatment
        for treatment in treatments_list:
            if rg in treatment.resources:
                num_resources = max(
                    num_resources,
                    treatment.resources[rg] if rg in treatment.resources else 0,
                )

        resources_needed[rgid] = num_resources

    # Now, for each resource group, generate resources and distribute total unavailability time with variability
    resources_dict = {}
    for rgid, num_resources in resources_needed.items():
        rg = resource_groups[rgid]
        unavailability_percentage = rg.unavailability_percentage
        total_unavailability_time_group = (
            total_work_time_per_resource * num_resources * unavailability_percentage
        )

        # Generate initial random unavailability times for each resource
        u_i_initial = []
        for _ in range(num_resources):
            # Assign a random unavailability time between 0 and 2 * group's unavailability percentage
            random_percentage = random.uniform(
                0, min(1.0, 2 * unavailability_percentage)
            )
            u_i = total_work_time_per_resource * random_percentage
            u_i_initial.append(u_i)

        # Scale unavailability times so that their sum matches the group's total unavailability time
        sum_u_initial = sum(u_i_initial)
        if sum_u_initial == 0:
            scaled_unavailability_times = [
                total_unavailability_time_group / num_resources
            ] * num_resources
        else:
            scaled_unavailability_times = [
                u_i * (total_unavailability_time_group / sum_u_initial)
                for u_i in u_i_initial
            ]

        # Generate resources with unavailability slots
        for i in range(num_resources):
            rid = next_rid
            resource_unavailability_time = scaled_unavailability_times[i]

            # Generate unavailability slots for this resource using the updated function
            unavailability_slots = generate_unavailability_slots(
                resource_unavailability_time,
                unavailability_duration_range,
                time_granularity_minutes,
                workday_start,
                workday_end,
                total_days,
            )
            resource = Resource(
                rid=rid,
                resource_group=rg,
                name=f"{rg.name}_{rid}",
                unavailable_time_slots=[],  # unavailability_slots,
            )
            resources_dict[rid] = resource
            next_rid += 1

    # Now that resources are generated, update patients to assign resource loyalties
    for patient in patients_dict.values():
        if patient.already_admitted:
            already_resource_loyal = {}
            for treatment, rg in patient.treatments_with_loyalty:
                num_resources_needed = treatment.resources[rg]
                loyal_resources = random.sample(
                    [
                        res
                        for res in resources_dict.values()
                        if res.resource_group == rg
                    ],
                    num_resources_needed,
                )
                already_resource_loyal[(treatment, rg)] = loyal_resources
            patient.already_resource_loyal = already_resource_loyal
            # Remove the temporary attribute
            del patient.treatments_with_loyalty

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
        resources=resources_dict,
        patients=patients_dict,
    )

    return instance
