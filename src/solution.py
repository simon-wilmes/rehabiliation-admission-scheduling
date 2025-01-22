from __future__ import annotations
from src.instance import Instance
from src.treatments import Treatment
from src.resource import ResourceGroup, Resource
from src.patients import Patient
from src.time import DayHour
from src.logging import logger
import math
from typing import Type
from src.utils import slice_dict
from collections import defaultdict
from math import ceil, floor
import hashlib

# Define the return code for the solver when no solution is found
NO_SOLUTION_FOUND = 0


class Appointment:
    def __init__(
        self,
        patients: list[Patient],
        start_date: DayHour,
        treatment: Treatment,
        resources: dict[ResourceGroup, list[Resource]],
        solver=None,  # type: ignore
    ):
        self.patients = patients
        self.start_date = start_date
        self.treatment = treatment
        self.resources = resources
        self.solver = solver
        self.is_valid_appointment()

    def is_valid_appointment(self):

        # Check that all resource groups required for the treatment are satisfied
        for required_group, amount in self.treatment.resources.items():
            if required_group not in self.resources:
                logger.warning(
                    f"Resource group {required_group.id} required for treatment {self.treatment.id} is missing "
                    f"at day {self.start_date.day}, hour {self.start_date.hour}."
                )
            if len(set(self.resources[required_group])) != amount:
                logger.warning(
                    f"Resource group {required_group.id} requires {amount} resources, "
                    f"but {len(set(self.resources[required_group]))} were provided "
                    f"at day {self.start_date.day}, hour {self.start_date.hour}."
                )

        # Check that no more than the maximum number of patients take part in this treatment
        if len(self.patients) > self.treatment.max_num_participants:
            logger.warning(
                f"Appointment has {len(self.patients)} patients, but the maximum is {self.treatment.max_num_participants} "
                f"at day {self.start_date.day}, hour {self.start_date.hour}."
            )
            # Check that no more than the maximum number of patients take part in this treatment

        if len(self.patients) < self.treatment.min_num_participants:
            logger.warning(
                f"Appointment has {len(self.patients)} patients, but the minimum is {self.treatment.min_num_participants} "
                f"at day {self.start_date.day}, hour {self.start_date.hour}."
            )

    def __le__(self, other):
        return self.start_date <= other.start_date

    def __lt__(self, other):
        return self.start_date < other.start_date

    def __repr__(self):
        return f"Appointment({self.start_date}, {self.patients},{self.treatment}, {self.resources})"


class Solution:

    def __init__(
        self,
        instance: Instance,
        schedule: list[Appointment],
        patients_arrival: dict[Patient, DayHour],
        solver=None,  # type: ignore
        solution_value: float = 0,
    ):
        self.instance = instance
        self.schedule = schedule  # List of Appointments
        self.patients_arrival = patients_arrival  # Dict[Patient, DayHour]
        self.solver = solver

        # remove all empty appintments
        self.schedule = list(filter(lambda x: len(x.patients), self.schedule))

        # Log the schedule for all patients and treatments
        for patients in self.patients_arrival:
            logger.debug(
                f"Patient {patients.id} is admitted at day {self.patients_arrival[patients].day}."
            )

        for appointment in self.schedule:
            resources = []
            for resource_group, resource_list in appointment.resources.items():
                resources.extend(resource_list)
            logger.debug(
                f"Patients {list(patient.id for patient in appointment.patients)} have treatment {appointment.treatment.id} "
                f"starting at day {appointment.start_date.day}, hour {appointment.start_date.hour} using resources {[f.id for f in resources]}."
            )

        # Perform the checks
        self._check_constraints()

        # Check
        if self.solver is not None:
            logger.debug("Checking solution value.")
            logger.debug("Solvers objective function value: %s", solution_value)

            self.value = self.calc_objective()
            logger.debug("Correct objective function value: %s", self.value)
            assert (
                solution_value == self.value
            ), "Solution value does not match calculated value."

            logger.debug("Solution is valid.")

    def _print_hash(self):
        def hash_appointments(appointments):
            appointments_str = sorted(
                f"{appt.start_date.day}-{appt.start_date.hour}-{appt.treatment.id}-"
                f"{sorted(p.id for p in appt.patients)}-"
                f"{sorted((rg.id, sorted(r.id for r in res)) for rg, res in appt.resources.items())}"
                for appt in appointments
            )
            return hashlib.md5("".join(appointments_str).encode()).hexdigest()

        def hash_patients_arrival(patients_arrival):
            arrival_str = sorted(
                f"{patient.id}-{dayhour.day}-{dayhour.hour}"
                for patient, dayhour in patients_arrival.items()
            )
            return hashlib.md5("".join(arrival_str).encode()).hexdigest()

        schedule_hash = hash_appointments(self.schedule)
        patients_arrival_hash = hash_patients_arrival(self.patients_arrival)
        solver_hash = (
            hashlib.md5(str(self.solver).encode()).hexdigest()
            if self.solver
            else "None"
        )

        logger.debug(f"Schedule Hash: {schedule_hash}")
        logger.debug(f"Patients Arrival Hash: {patients_arrival_hash}")
        logger.debug(f"Solver Hash: {solver_hash}")
        logger.debug(
            f"Solution Hash: {hashlib.md5((schedule_hash + patients_arrival_hash + solver_hash).encode()).hexdigest()}"
        )

    def check_other_solvers(self):
        from src.solvers import CPSolver, CPSolver2, MIPSolver, MIPSolver2, Solver
        import logging

        if logger.getEffectiveLevel() < logging.WARNING:
            prev_level = logger.getEffectiveLevel()
            logger.setLevel("WARNING")
        solvers_cls: list[Type[Solver]] = []
        best_result = True

        for solver_cls in solvers_cls:
            solver = solver_cls(
                self.instance,
                break_symetry=True,
                log_to_console=False,
            )

            solver.create_model()
            solver.assert_solution(self)
            try:
                solution = solver.solve_model(check_better_solution=False)
                if solution is NO_SOLUTION_FOUND:
                    logger.warning(
                        "The Solver %s could not find the same solution as Solver %s",
                        solver_cls.__name__,
                        self.solver.__class__.__name__,
                    )
                assert (
                    solution.value == self.value
                ), "The solution value %f is different for solver %s with %f " % (
                    solver_cls.__name__,
                    self.value,
                    solution.value,
                )

            except Exception as e:
                logger.error(
                    "There was an error during verifying the value of solution by solver %s",
                    solver_cls.__name__,
                )
                logger.error(e)
        logger.setLevel(prev_level)
        logger.debug("All solvers accepted the solution.")

    def calc_objective(self):
        """
        Calculate the objective function value for this solution.
        """
        from src.solvers import Solver

        assert isinstance(self.solver, Solver), "Solver is not set."

        # Calculate the treatment value
        treatment_obj = self.solver.treatment_value * len(self.schedule)
        # Calculate the delay value
        delay_obj = self.solver.delay_value * sum(
            self.patients_arrival[p].day - p.earliest_admission_date.day
            for p in self.instance.patients.values()
        )

        missing_obj = 0
        # calculate the missing treatment value

        scheduled_treatments = defaultdict(int)
        for appointment in self.schedule:
            for patient in appointment.patients:
                scheduled_treatments[patient] += 1

        horizon_length = self.instance.horizon_length
        for patient in self.instance.patients.values():
            # calc how many treatments should be scheduled

            num_treatments = sum(
                value
                for value in slice_dict(self.solver.lr_pm, (patient, None)).values()  # type: ignore
            )

            scheduled_count = scheduled_treatments.get(patient, 0)

            missing_obj += (
                max(0, num_treatments - scheduled_count)
                * self.solver.missing_treatment_value
            )

        logger.debug("(VERIFIER):Treatment value: %s", treatment_obj)
        logger.debug("(VERIFIER):Delay value: %s", delay_obj)
        logger.debug("(VERIFIER):Missing treatment value: %s", missing_obj)
        return treatment_obj + delay_obj + missing_obj

    def _check_constraints(self):
        self._check_patient_admission()
        self._check_treatment_assignment_during_stay()
        self._check_no_overlapping_appointments()
        self._check_resource_availability_and_uniqueness()

        self._check_bed_capacity()
        self._check_total_treatments_scheduled()
        self._check_no_treatments_outside_horizont()
        self._check_daily_scheduling()

        self._check_even_scheduling()
        self._print_resource_usage()
        self._print_hash()

    def _check_patient_admission(self):
        """
        Check that each patient is admitted exactly once within their earliest and latest admission dates,
        considering their length of stay.
        """
        for patient in self.instance.patients.values():
            if patient not in self.patients_arrival:
                logger.warning(f"Patient {patient.id} does not have an admission date.")

            admission_day = self.patients_arrival[patient].day
            earliest_admission = patient.earliest_admission_date.day
            latest_admission = patient.admitted_before_date.day

            if not (earliest_admission <= admission_day < latest_admission):
                logger.warning(
                    f"Patient {patient.id} admission day {admission_day} is outside "
                    f"their admission window ({earliest_admission} to {latest_admission})."
                )

    def _check_treatment_assignment_during_stay(self):
        """
        Ensure that for every treatment all schedules lie inside the patient's stay.
        """
        admitted_patients = set(self.patients_arrival.keys())

        for appointment in self.schedule:
            for patient in appointment.patients:
                if (
                    self.patients_arrival[patient].day > appointment.start_date.day
                    or self.patients_arrival[patient].day + patient.length_of_stay
                    <= appointment.start_date.day
                ):
                    logger.warning(
                        f"Patient {patient.id} has treatments scheduled during day {appointment.start_date.day} but was admitted on day {self.patients_arrival[patient]} and has a length of stay of {patient.length_of_stay}."
                    )

    def _check_no_overlapping_appointments(self):
        """
        Ensure that no patient has overlapping appointments.
        """
        # For each patient, collect all their appointments with start and end times
        patient_appointments = {}
        for patient in self.instance.patients.values():
            patient_appointments[patient] = []

        for appointment in self.schedule:
            start_day = appointment.start_date.day
            start_time = appointment.start_date.hour
            duration = appointment.treatment.duration.hours  # Duration object has .hour
            end_time = start_time + duration
            for patient in appointment.patients:
                patient_appointments[patient].append(
                    (start_day, start_time, end_time, appointment)
                )

        for patient, appts in patient_appointments.items():
            # Sort appointments by start time
            appts.sort()
            for i in range(len(appts)):
                day_i, start_i, end_i, appt_i = appts[i]
                for j in range(i + 1, len(appts)):
                    day_j, start_j, end_j, appt_j = appts[j]
                    if day_i != day_j:
                        continue  # Appointments are on different days
                    # Check for overlap
                    if start_j < end_i and end_j > start_i:
                        logger.warning(
                            f"Patient {patient.id} has overlapping appointments "
                            f"on day {day_i}: {appt_i.treatment.id} ({start_i}-{end_i}) "
                            f"and {appt_j.treatment.id} ({start_j}-{end_j})."
                        )

    def _check_resource_availability_and_uniqueness(self):
        """
        Ensure resources are not double-booked, are only assigned when available,
        and that resource lists in appointments do not contain duplicates.
        """
        resource_usage = {}  # Key: (resource.id, day, time), Value: appointment
        for appointment in self.schedule:
            start_day = appointment.start_date.day
            start_time = appointment.start_date.hour
            duration = appointment.treatment.duration.hours
            end_time = start_time + duration
            # For each resource group, get the list of resources
            for resource_group, resources in appointment.resources.items():
                # Check for duplicates in resources
                resource_ids = [res.id for res in resources]
                if len(resource_ids) != len(set(resource_ids)):
                    logger.warning(
                        f"Appointment starting at day {start_day}, time {start_time} "
                        f"has duplicate resources in resource group {resource_group.id}."
                    )
                for resource in resources:
                    # Check resource availability and double-booking
                    time = start_time
                    while time < end_time:
                        key = (resource.id, start_day, time)
                        if key in resource_usage:
                            other_appointment = resource_usage[key]
                            logger.warning(
                                f"Resource {resource.id} is double-booked at day {start_day}, time {time} "
                                f"between appointments for treatments {appointment.treatment.id} "
                                f"and {other_appointment.treatment.id}."
                            )
                        # Check resource availability
                        if (
                            not resource.is_available(DayHour(day=start_day, hour=time))
                            or time > self.instance.workday_end.hour
                            or time < self.instance.workday_start.hour
                        ):
                            logger.warning(
                                f"Resource {resource.id} is not available at day {start_day}, time {time}."
                            )
                        resource_usage[key] = appointment
                        # Move to next time slot
                        time += (
                            self.instance.time_slot_length.hours
                        )  # Assuming time_slot_length is a Duration object

    def _check_resource_loyalty(self):
        """
        For treatments requiring resource loyalty, confirm that each patient uses the same resource
        for all instances of that treatment and resource group, taking into account already_resource_loyal.
        """
        # Build a dictionary for each patient, treatment, resource group, to track resources used
        patient_resource_loyalty = {}

        for appointment in self.schedule:
            treatment = appointment.treatment
            for patient in appointment.patients:
                for resource_group, resources in appointment.resources.items():
                    # Check if the treatment requires loyalty for this resource group
                    required_amount = treatment.resources.get(resource_group)
                    if required_amount is None:
                        continue  # Resource group not required for this treatment
                    loyalty_flag = treatment.loyalty.get(resource_group, False)  # type: ignore
                    if not loyalty_flag:
                        continue  # Resource loyalty not required for this resource group

                    key = (patient, treatment, resource_group)
                    # Get the list of resource IDs
                    resource_ids = tuple(sorted([res.id for res in resources]))
                    if key in patient_resource_loyalty:
                        if patient_resource_loyalty[key] != resource_ids:
                            logger.warning(
                                f"Patient {patient.id} has inconsistent resource usage for "
                                f"treatment {treatment.id}, resource group {resource_group.id}. "
                                f"Previously used resources {patient_resource_loyalty[key]}, "
                                f"now uses {resource_ids}."
                            )
                    else:
                        # Check if already_resource_loyal is set
                        if hasattr(patient, "already_resource_loyal"):
                            arl_key = (treatment, resource_group)
                            if arl_key in patient.already_resource_loyal:
                                existing_resources = patient.already_resource_loyal[
                                    arl_key
                                ]
                                existing_resource_ids = tuple(
                                    sorted([res.id for res in existing_resources])
                                )
                                if resource_ids != existing_resource_ids:
                                    logger.warning(
                                        f"Patient {patient.id} has resource loyalty constraint to resources "
                                        f"{existing_resource_ids} for treatment {treatment.id}, resource group {resource_group.id}, "
                                        f"but uses resources {resource_ids}."
                                    )
                        patient_resource_loyalty[key] = resource_ids

    def _check_bed_capacity(self):
        """
        Ensure the total number of admitted patients per day does not exceed the number of beds.
        """
        bed_capacity = self.instance.beds_capacity  # b
        # For each day, count the number of patients admitted
        admitted_patients = {}  # Key: day, Value: set of patients
        for patient, arrival_dayhour in self.patients_arrival.items():
            length_of_stay = patient.length_of_stay  # l_p
            for day in range(arrival_dayhour.day, arrival_dayhour.day + length_of_stay):
                if day not in admitted_patients:
                    admitted_patients[day] = set()
                admitted_patients[day].add(patient)
        for day, patients in admitted_patients.items():
            if self.solver is not None and day in self.solver.D:
                if len(patients) > bed_capacity:
                    logger.warning(
                        f"Day {day} has {len(patients)} patients admitted, which exceeds bed capacity ({bed_capacity})."
                    )

    def _check_total_treatments_scheduled(self):
        """
        Ensure the total number of scheduled treatments equals the total repetitions left
        for each patient and treatment.
        """
        # For each patient and treatment, count the number of times it's scheduled
        scheduled_treatments = {}
        for appointment in self.schedule:
            treatment = appointment.treatment
            for patient in appointment.patients:
                key = (patient.id, treatment.id)
                if key not in scheduled_treatments:
                    scheduled_treatments[key] = 0
                scheduled_treatments[key] += 1
        # Now check against lr_pm which is the total repetitions left
        for patient in self.instance.patients.values():
            for treatment in patient.treatments.keys():
                try:
                    lr_pm = (
                        patient.treatments[treatment]
                        - patient.already_scheduled_treatments[treatment]
                    )  # Total repetitions left
                except KeyError:
                    pass
                key = (patient.id, treatment.id)
                scheduled_count = scheduled_treatments.get(key, 0)
                if scheduled_count > lr_pm:
                    logger.warning(
                        f"Patient {patient.id} has {scheduled_count} scheduled treatments for treatment {treatment.id}, "
                        f"but needs {lr_pm} out of the initial {patient.treatments[treatment]} repetitions."
                    )

    def _check_conflict_groups(self):
        """
        Ensure that for any patient, treatments that are in the same conflict group are not scheduled on the same day.
        """

        # Build a mapping of patient to a dict of day to set of treatments
        patient_day_treatments = {}  # patient -> day -> set(treatments)
        for appointment in self.schedule:
            day = appointment.start_date.day
            for patient in appointment.patients:
                if patient not in patient_day_treatments:
                    patient_day_treatments[patient] = {}
                if day not in patient_day_treatments[patient]:
                    patient_day_treatments[patient][day] = set()
                patient_day_treatments[patient][day].add(appointment.treatment)

        # For each patient, day, check conflict groups
        for patient, day_treatments in patient_day_treatments.items():
            for day, treatments in day_treatments.items():
                for conflict_group in self.instance.conflict_groups:
                    treatments_in_group = treatments.intersection(conflict_group)
                    if len(treatments_in_group) > 1:
                        conflict_treatment_ids = [
                            treatment.id for treatment in treatments_in_group
                        ]
                        logger.warning(
                            f"Patient {patient.id} has treatments {conflict_treatment_ids} from the same conflict group scheduled on day {day}."
                        )

    def _check_even_scheduling(self):
        """
        For every patient, ensure that in any rolling window, the number of scheduled treatments
        is less than or equal to the average number of treatments that this patient should have received
        during this time frame considering the total treatments and the patient's length of stay, rounded up.
        """

        # List[Tuple[int, int]]  # List of (start_day, window_length)
        # For each patient
        patient_days = defaultdict(int)

        for app in self.schedule:
            for patient in app.patients:
                patient_days[patient, app.start_date.day] += 1

        e_w = self.instance.even_scheduling_width
        e_ub = self.instance.even_scheduling_upper

        for patient in self.instance.patients.values():
            avg_treatments = (
                sum(n_m for n_m in patient.treatments.values())
                * e_w
                / patient.length_of_stay
            )

            for day in range(
                patient.earliest_admission_date.day,
                patient.admitted_before_date.day + patient.length_of_stay + 1,
            ):
                treatments_scheduled = sum(
                    patient_days[patient, d] for d in range(day, day + e_w)
                )

                # Check upperbound
                if treatments_scheduled > ceil(avg_treatments * e_ub):
                    logger.warning(
                        f"Patient {patient.id} has {treatments_scheduled} treatments scheduled in the window "
                        f"starting at day {day}, which exceeds the upper bound of {ceil(avg_treatments * e_ub)}."
                    )

    def _check_no_treatments_outside_horizont(self):
        """
        Ensure that no treatment is scheduled outside the horizon length.
        """
        horizon_length = self.instance.horizon_length
        for appointment in self.schedule:
            start_day = appointment.start_date.day
            if start_day >= horizon_length:
                logger.warning(
                    f"Appointment for treatment {appointment.treatment.id} is scheduled outside the horizon length."
                )

    def _check_rest_times(self):
        """
        Ensure that each patient respects the treatment's rest time between consecutive treatments.
        After finishing a treatment, the patient must wait treatment.rest_time.hours
        before starting another treatment.
        """
        patient_appointments = defaultdict(list)
        for appointment in self.schedule:
            start_day = appointment.start_date.day
            start_time = appointment.start_date.hour
            duration = appointment.treatment.duration.hours
            end_time = start_time + duration
            for patient in appointment.patients:
                patient_appointments[patient].append(
                    (start_day, start_time, end_time, appointment)
                )

        for patient, appts in patient_appointments.items():
            # Sort appointments chronologically
            appts.sort(key=lambda x: (x[0], x[1]))
            for i in range(len(appts) - 1):
                _, _, end_i, appt_i = appts[i]
                next_day, next_start, _, appt_j = appts[i + 1]

                # Convert start/end times to absolute hours from day 0 for easier comparison
                abs_end_i = appts[i][0] * 24 + end_i
                abs_start_j = next_day * 24 + next_start

                required_rest = appt_i.treatment.rest_time.hours

                if abs_start_j < abs_end_i + required_rest:
                    logger.warning(
                        f"Patient {patient.id} has appointments too close together. After finishing treatment "
                        f"{appt_i.treatment.id} at day {appts[i][0]}, hour {end_i}, patient must rest {required_rest} "
                        f"hours before another treatment. The next treatment {appt_j.treatment.id} starts at day {next_day}, "
                        f"hour {next_start}, which is too early."
                    )

    def _check_daily_scheduling(self):
        """
        For every patient and every day of their stay, ensure that the daily number of treatments
        is between floor(avg_per_day * daily_scheduling_lower) and ceil(avg_per_day * daily_scheduling_upper).

        avg_per_day = total required treatments / length_of_stay
        """
        # Count treatments per patient per day
        patient_day_count = defaultdict(int)
        for appointment in self.schedule:
            for patient in appointment.patients:
                patient_day_count[(patient, appointment.start_date.day)] += 1

        # Parameters from the instance
        daily_upper = self.instance.daily_scheduling_upper
        daily_lower = self.instance.daily_scheduling_lower

        for patient in self.instance.patients.values():
            total_required_treatments = sum(
                patient.treatments[t] - patient.already_scheduled_treatments.get(t, 0)
                for t in patient.treatments
            )
            avg_per_day = total_required_treatments / patient.length_of_stay

            for day in range(
                self.patients_arrival[patient].day,
                min(
                    self.patients_arrival[patient].day + patient.length_of_stay,
                    self.instance.horizon_length,
                ),
            ):
                treatments_today = patient_day_count.get((patient, day), 0)
                upper_bound = math.ceil(avg_per_day * daily_upper)
                lower_bound = math.floor(avg_per_day * daily_lower)

                if treatments_today > upper_bound or treatments_today < lower_bound:
                    logger.warning(
                        f"Patient {patient.id} on day {day} has {treatments_today} treatments, "
                        f"expected {lower_bound} <= x <={upper_bound}   (avg={avg_per_day:.2f}, "
                        f"multipliers={daily_upper}/{daily_lower})."
                    )

    def _print_resource_usage(self):
        """
        Calculate resource utilization by comparing assigned time to available time.
        Returns a dictionary with resource utilization factors.
        """
        if self.solver is None:
            logger.warning("Solver is not set, cannot calculate resource utilization.")
            return
        resource_usage = defaultdict(float)  # Track assigned hours per resource
        resource_availability = defaultdict(float)  # Track available hours per resource

        # Calculate total available hours for each resource
        for day in range(self.instance.horizon_length):
            for resource in self.instance.resources.values():
                resource_availability[resource.id] += resource.total_availability_hours(
                    day
                )

        # Calculate assigned hours for each resource
        for appointment in self.schedule:
            duration = appointment.treatment.duration.hours
            for _, resources in appointment.resources.items():
                for resource in resources:
                    resource_usage[resource.id] += duration

        # Calculate utilization factor for each resource
        utilization_factors = {}
        for resource_id in resource_availability.keys():
            if resource_availability[resource_id] > 0:
                utilization_factors[resource_id] = (
                    resource_usage[resource_id] / resource_availability[resource_id]
                )
            else:
                utilization_factors[resource_id] = 0
                logger.warning(f"Resource {resource_id} has no available hours")

        logger.debug("Resource utilization factors:")
        for resource_id, factor in utilization_factors.items():
            logger.debug(f"Resource {resource_id}: {factor:.2f}")
        logger.debug(
            "Average utilization factor: %.2f",
            sum(utilization_factors.values()) / len(utilization_factors),
        )
