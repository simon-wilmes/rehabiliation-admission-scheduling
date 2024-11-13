from src.instance import Instance
from src.treatments import Treatment
from src.resource import ResourceGroup, Resource
from src.patients import Patient
from src.time import DayHour
from src.logging import logger


class Appointment:
    def __init__(
        self,
        patients: list[Patient],
        start_date: DayHour,
        treatment: Treatment,
        resources: dict[ResourceGroup, list[Resource]],
    ):
        self.patients = patients
        self.start_date = start_date
        self.treatment = treatment
        self.resources = resources
        assert self.is_valid_appointment()

    def is_valid_appointment(self):
        # Check that all resource groups required for the treatment are satisfied
        for required_group, amount in self.treatment.resources.items():
            if required_group not in self.resources:
                return False
            if len(set(self.resources[required_group])) != amount:
                return False

        return True

    def __repr__(self):
        return f"Appointment({self.start_date}, {self.patients},{self.treatment}, {self.resources})"


class Solution:
    def __init__(self, instance, schedule, patients_arrival):
        self.instance = instance
        self.schedule = schedule  # List of Appointments
        self.patients_arrival = patients_arrival  # Dict[Patient, DayHour]

        # Perform the checks
        self._check_constraints()
        logger.info("Solution is valid.")

    def _check_constraints(self):
        self._check_patient_admission()
        self._check_treatment_assignment()
        self._check_no_overlapping_appointments()
        self._check_resource_availability_and_uniqueness()
        self._check_resource_loyalty()
        self._check_max_patients_per_treatment()
        self._check_bed_capacity()
        self._check_total_treatments_scheduled()

    def _check_patient_admission(self):
        """
        Check that each patient is admitted exactly once within their earliest and latest admission dates,
        considering their length of stay.
        """
        for patient in self.instance.patients.values():
            if patient not in self.patients_arrival:
                raise ValueError(
                    f"Patient {patient.id} does not have an admission date."
                )

            admission_day = self.patients_arrival[patient].day
            earliest_admission = patient.earliest_admission_date.day
            latest_admission = patient.latest_admission_date.day

            if not (earliest_admission <= admission_day <= latest_admission):
                raise ValueError(
                    f"Patient {patient.id} admission day {admission_day} is outside "
                    f"their admission window ({earliest_admission} to {latest_admission})."
                )

    def _check_treatment_assignment(self):
        """
        Ensure that patients not admitted have no treatments scheduled.
        """
        admitted_patients = set(self.patients_arrival.keys())

        for appointment in self.schedule:
            for patient in appointment.patients:
                if patient not in admitted_patients:
                    raise ValueError(
                        f"Patient {patient.id} has treatments scheduled but is not admitted."
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
            duration = (
                appointment.treatment.duration.hours
            )  # Duration object has .hours
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
                        raise ValueError(
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
                    raise ValueError(
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
                            raise ValueError(
                                f"Resource {resource.id} is double-booked at day {start_day}, time {time} "
                                f"between appointments for treatments {appointment.treatment.id} "
                                f"and {other_appointment.treatment.id}."
                            )
                        # Check resource availability
                        if not resource.is_available(DayHour(day=start_day, hour=time)):
                            raise ValueError(
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
                    loyalty_flag = treatment.loyalty.get(resource_group, False)
                    if not loyalty_flag:
                        continue  # Resource loyalty not required for this resource group

                    key = (patient, treatment, resource_group)
                    # Get the list of resource IDs
                    resource_ids = tuple(sorted([res.id for res in resources]))
                    if key in patient_resource_loyalty:
                        if patient_resource_loyalty[key] != resource_ids:
                            raise ValueError(
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
                                    raise ValueError(
                                        f"Patient {patient.id} has resource loyalty constraint to resources "
                                        f"{existing_resource_ids} for treatment {treatment.id}, resource group {resource_group.id}, "
                                        f"but uses resources {resource_ids}."
                                    )
                        patient_resource_loyalty[key] = resource_ids

    def _check_max_patients_per_treatment(self):
        """
        Ensure the maximum number of patients per treatment is not exceeded.
        """
        # For each treatment, day, time, count the number of patients
        treatment_schedule = {}
        for appointment in self.schedule:
            treatment = appointment.treatment
            start_day = appointment.start_date.day
            start_time = appointment.start_date.hour
            key = (treatment.id, start_day, start_time)
            if key not in treatment_schedule:
                treatment_schedule[key] = []
            treatment_schedule[key].extend(appointment.patients)

        for (treatment_id, day, time), patients in treatment_schedule.items():
            treatment = self.instance.treatments[treatment_id]
            max_patients = treatment.num_participants  # k_m
            if len(patients) > max_patients:
                raise ValueError(
                    f"Treatment {treatment_id} at day {day}, time {time} has {len(patients)} patients "
                    f"scheduled, which exceeds the maximum allowed ({max_patients})."
                )

    def _check_bed_capacity(self):
        """
        Ensure the total number of admitted patients per day does not exceed the number of beds.
        """
        bed_capacity = self.instance.num_beds  # b
        # For each day, count the number of patients admitted
        admitted_patients = {}  # Key: day, Value: set of patients
        for patient, arrival_dayhour in self.patients_arrival.items():
            length_of_stay = patient.length_of_stay  # l_p
            for day in range(arrival_dayhour.day, arrival_dayhour.day + length_of_stay):
                if day not in admitted_patients:
                    admitted_patients[day] = set()
                admitted_patients[day].add(patient)
        for day, patients in admitted_patients.items():
            if len(patients) > bed_capacity:
                raise ValueError(
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
        # Now check against r_pm
        for patient in self.instance.patients.values():
            for treatment in patient.treatments.keys():
                r_pm = patient.treatments[treatment]  # Total repetitions left
                key = (patient.id, treatment.id)
                scheduled_count = scheduled_treatments.get(key, 0)
                if scheduled_count != r_pm:
                    raise ValueError(
                        f"Patient {patient.id} has {scheduled_count} scheduled treatments for treatment {treatment.id}, "
                        f"but needs {r_pm} repetitions."
                    )
