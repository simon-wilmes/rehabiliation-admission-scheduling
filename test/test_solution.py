import pytest
from src.solution import Solution, Appointment
from src.instance import Instance
from src.patients import Patient
from src.treatments import Treatment
from src.resource import Resource, ResourceGroup
from src.time import DayHour, Duration
from collections import defaultdict


@pytest.fixture
def setup_instance():
    # Create resources
    rg_therapists = ResourceGroup(0, "therapists")
    rg_rooms = ResourceGroup(1, "rooms")
    therapist0 = Resource(0, rg_therapists, "therapist0")
    therapist1 = Resource(1, rg_therapists, "therapist1")
    room0 = Resource(3, rg_rooms, "room0")
    room1 = Resource(4, rg_rooms, "room1")

    # Create treatments
    treatment0 = Treatment(
        tid=0,
        num_participants=2,
        duration=Duration(hours=1),
        name="therapy",
        resources={rg_therapists: (1, True), rg_rooms: (1, False)},
    )
    treatment1 = Treatment(
        tid=1,
        num_participants=1,
        duration=Duration(hours=1),
        name="counseling",
        resources={rg_therapists: (1, True), rg_rooms: (1, False)},
    )

    # Create patients
    patient0 = Patient(
        pid=0,
        treatments={treatment0: 2},
        length_of_stay=10,
        earliest_admission_date=DayHour(day=0, hour=0),
        admitted_before_date=DayHour(day=10, hour=0),
        already_admitted=False,
        already_resource_loyal={(treatment0, rg_therapists): [therapist0]},
        name="patient0",
    )

    instance_data = {
        "num_beds": 2,
        "workday_start": DayHour(hour=8),
        "workday_end": DayHour(hour=17),
        "rolling_window_length": 7,
        "rolling_windows_days": [0, 5, 10],
        "conflict_groups": [(0, 1)],
        "time_slot_length": Duration(hours=0, minutes=15),
    }

    instance = Instance(
        instance_data=instance_data,
        resource_groups={0: rg_therapists, 1: rg_rooms},
        treatments={0: treatment0, 1: treatment1},
        resources={0: therapist0, 1: therapist1, 3: room0, 4: room1},
        patients={0: patient0},
    )

    return (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    )


def test_already_scheduled_treatments(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance

    patient0.treatments = {treatment0: 2}
    # Assume patient0 already has one treatment scheduled
    patient0.already_scheduled_treatments = {
        treatment0: 1,
    }

    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=2, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
    ]

    # Should raise an error
    with pytest.raises(ValueError, match="needs .* repetitions"):
        Solution(instance, schedule, patients_arrival)

    # Now remove one existing treatment
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        )
    ]

    solution = Solution(instance, schedule, patients_arrival)
    assert solution is not None

    # Now remove the already scheduled treatment
    patient0.already_scheduled_treatments = defaultdict(int)
    # Should raise an error
    with pytest.raises(ValueError, match="needs .* repetitions"):
        Solution(instance, schedule, patients_arrival)


def test_patient_admission(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance
    patients_arrival = {patient0: DayHour(day=11, hour=0)}  # Invalid admission date
    schedule = []
    with pytest.raises(
        ValueError, match="admission day .* is outside their admission window"
    ):
        Solution(instance, schedule, patients_arrival)


def test_treatment_assignment(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance
    patients_arrival = {}
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        )
    ]
    with pytest.raises(ValueError, match="Patient .* does not have an admission date"):
        Solution(instance, schedule, patients_arrival)


def test_no_overlapping_appointments(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance
    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
    ]
    with pytest.raises(ValueError, match="has overlapping appointments"):
        Solution(instance, schedule, patients_arrival)


def test_resource_availability_and_uniqueness(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance
    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0, therapist0], rg_rooms: [room0]},
        )
    ]
    with pytest.raises(ValueError, match="has duplicate resources in resource group"):
        Solution(instance, schedule, patients_arrival)


def test_resource_loyalty(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance
    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist1], rg_rooms: [room0]},
        )
    ]
    with pytest.raises(
        ValueError, match="Patient [0-9]* has resource loyalty constraint to resources"
    ):
        Solution(instance, schedule, patients_arrival)


def test_max_and_min_patients_per_treatment(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance
    patient1 = Patient(
        pid=1,
        treatments={treatment0: 1},
        length_of_stay=10,
        earliest_admission_date=DayHour(day=0, hour=0),
        admitted_before_date=DayHour(day=10, hour=0),
        name="patient1",
    )
    patient2 = Patient(
        pid=2,
        treatments={treatment0: 1},
        length_of_stay=10,
        earliest_admission_date=DayHour(day=0, hour=0),
        admitted_before_date=DayHour(day=10, hour=0),
        name="patient2",
    )
    instance.patients[1] = patient1
    instance.patients[2] = patient2
    patients_arrival = {
        patient0: DayHour(day=0, hour=0),
        patient1: DayHour(day=0, hour=0),
        patient2: DayHour(day=0, hour=0),
    }

    with pytest.raises(
        ValueError, match="Appointment has [0-9]* patients, but the maximum is"
    ):
        schedule = [
            Appointment(
                [patient0, patient1, patient2],
                DayHour(day=1, hour=9),
                treatment0,
                {rg_therapists: [therapist0], rg_rooms: [room0]},
            ),
            Appointment(
                [patient0],
                DayHour(day=2, hour=9),
                treatment0,
                {rg_therapists: [therapist0], rg_rooms: [room0]},
            ),
        ]
        Solution(instance, schedule, patients_arrival)

    # Test that empty appointment is found

    with pytest.raises(ValueError, match="Appointment has no patients at"):
        schedule = [
            Appointment(
                [patient0, patient1],
                DayHour(day=1, hour=9),
                treatment0,
                {rg_therapists: [therapist0], rg_rooms: [room0]},
            ),
            Appointment(
                [patient0],
                DayHour(day=2, hour=9),
                treatment0,
                {rg_therapists: [therapist0], rg_rooms: [room0]},
            ),
            Appointment(
                [patient2],
                DayHour(day=3, hour=9),
                treatment0,
                {rg_therapists: [therapist0], rg_rooms: [room0]},
            ),
            Appointment(
                [],
                DayHour(day=4, hour=9),
                treatment0,
                {rg_therapists: [therapist0], rg_rooms: [room0]},
            ),
        ]
        Solution(instance, schedule, patients_arrival)


def test_bed_capacity(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance
    instance.beds_capacity = 1
    patient1 = Patient(
        pid=1,
        treatments={treatment0: 2},
        length_of_stay=10,
        earliest_admission_date=DayHour(day=0, hour=0),
        admitted_before_date=DayHour(day=11, hour=0),
        name="patient1",
    )
    instance.patients[1] = patient1
    patients_arrival = {
        patient0: DayHour(day=0, hour=0),
        patient1: DayHour(day=9, hour=0),
    }
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=0, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient1],
            DayHour(day=11, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient1],
            DayHour(day=12, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
    ]
    with pytest.raises(ValueError, match="exceeds bed capacity"):
        Solution(instance, schedule, patients_arrival)

    patients_arrival = {
        patient0: DayHour(day=0, hour=0),
        patient1: DayHour(day=10, hour=0),
    }

    assert (
        Solution(instance, schedule, patients_arrival) is not None
    ), "Valid Solution was not accepted."


def test_patient_arrives_too_late(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance
    patients_arrival = {patient0: DayHour(day=10)}  # Arrives too late
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=11, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=12, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
    ]
    with pytest.raises(
        ValueError, match="admission day .* is outside their admission window"
    ):
        Solution(instance, schedule, patients_arrival)

    patients_arrival = {patient0: DayHour(day=9)}  # Arrives just in time
    assert (
        Solution(instance, schedule, patients_arrival) is not None
    ), "Valid Solution was not accepted."


def test_total_treatments_scheduled(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance
    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        )
    ]
    with pytest.raises(ValueError, match="needs .* repetitions"):
        Solution(instance, schedule, patients_arrival)


def test_valid_solution(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance
    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=2, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
    ]
    solution = Solution(instance, schedule, patients_arrival)
    assert solution is not None


def test_conflict_groups_violation(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance

    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    patient0.treatments = {treatment0: 1, treatment1: 1}
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=1, hour=10),  # Same day as previous appointment
            treatment1,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
    ]

    with pytest.raises(
        ValueError, match="from the same conflict group scheduled on day"
    ):
        Solution(instance, schedule, patients_arrival)


def test_conflict_groups_no_violation(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance

    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    patient0.treatments = {treatment0: 1, treatment1: 1}
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=2, hour=9),  # Different day
            treatment1,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
    ]

    # Should not raise any error
    solution = Solution(
        instance,
        schedule,
        patients_arrival,
    )
    assert solution is not None


def test_conflict_groups_not_enforced(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance

    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    patient0.treatments = {treatment0: 1, treatment1: 1}
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=1, hour=10),  # Same day, conflicting treatments
            treatment1,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
    ]

    # Should not raise an error because constraint not enforced
    solution = Solution(
        instance,
        schedule,
        patients_arrival,
        ignored_constraints={
            "conflict_groups",
        },
    )
    assert solution is not None


def test_even_scheduling_violation(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance

    instance.rolling_window_length = 7
    instance.rolling_window_days = [0]

    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    patient0.treatments = {treatment0: 4, treatment1: 2}
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=2, hour=9),
            treatment1,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
        Appointment(
            [patient0],
            DayHour(day=3, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=4, hour=12),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=4, hour=9),
            treatment1,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
        Appointment(
            [patient0],
            DayHour(day=5, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
    ]

    with pytest.raises(ValueError, match="exceeds the expected .* treatments"):
        Solution(
            instance,
            schedule,
            patients_arrival,
        )


def test_even_scheduling_certain_days_violated(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance

    instance.rolling_window_length = 7
    instance.rolling_window_days = [0]
    patient0.treatments = {treatment0: 4}
    patients_arrival = {patient0: DayHour(day=0, hour=0)}

    # Schedule 2 treatments in first 7 days, which is acceptable
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=2, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=3, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=4, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=7, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
    ]

    # Should not raise any error
    solution = Solution(
        instance,
        schedule,
        patients_arrival,
    )
    assert solution is not None

    instance.rolling_window_length = 7
    instance.rolling_window_days = [2]

    with pytest.raises(ValueError, match="exceeds the expected .* treatments"):
        Solution(
            instance,
            schedule,
            patients_arrival,
        )


def test_even_scheduling_not_enforced(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance

    instance.rolling_window_length = 7
    instance.rolling_window_days = [0]

    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    patient0.treatments = {treatment0: 3, treatment1: 2}
    # Schedule 5 treatments in first 7 days, exceeding expected 4 treatments
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=2, hour=9),
            treatment1,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
        Appointment(
            [patient0],
            DayHour(day=3, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=4, hour=9),
            treatment1,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
        Appointment(
            [patient0],
            DayHour(day=5, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
    ]

    # Should not raise an error because constraint not enforced
    solution = Solution(
        instance, schedule, patients_arrival, ignored_constraints={"even_scheduling"}
    )
    assert solution is not None


def test_even_scheduling_integer_average(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance

    # Adjust patient length of stay to 10
    patient0.length_of_stay = 10
    total_treatments_p = 5
    average_per_day = total_treatments_p / patient0.length_of_stay

    patient0.treatments = {treatment0: total_treatments_p}
    instance.rolling_window_length = 6
    instance.rolling_window_days = [0]

    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    patient0.treatments = {treatment0: 5}
    # Schedule 4 treatments in first 6 days, exceeding expected 3 treatments
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=0, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=3, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=5, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=7, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
    ]

    with pytest.raises(ValueError, match="exceeds the expected .* treatments"):
        Solution(
            instance,
            schedule,
            patients_arrival,
        )
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=0, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=3, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=6, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=7, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
    ]

    # Now schedule only 3 treatments
    # Should be acceptable
    solution = Solution(
        instance,
        schedule,
        patients_arrival,
    )
    assert solution is not None


def test_even_scheduling_overlapping_windows(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance

    # Keep length of stay 10, total treatments 5
    instance.rolling_window_length = 7
    instance.rolling_window_days = [0, 5]  # Overlapping windows

    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    patient0.treatments = {treatment0: 3, treatment1: 2}
    # Schedule treatments spread across days
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=3, hour=9),
            treatment1,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
        Appointment(
            [patient0],
            DayHour(day=5, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=7, hour=9),
            treatment1,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
        Appointment(
            [patient0],
            DayHour(day=9, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
    ]

    # Should not raise any error
    solution = Solution(
        instance,
        schedule,
        patients_arrival,
    )
    assert solution is not None


def test_even_scheduling_overlapping_windows_violation(setup_instance):
    (
        instance,
        patient0,
        treatment0,
        treatment1,
        therapist0,
        therapist1,
        room0,
        room1,
        rg_therapists,
        rg_rooms,
    ) = setup_instance

    # Keep length of stay 10, total treatments 5
    instance.rolling_window_length = 6
    instance.rolling_window_days = [0, 5]  # Overlapping windows

    patients_arrival = {patient0: DayHour(day=0, hour=0)}
    patient0.treatments = {treatment0: 3, treatment1: 3}
    # Schedule treatments spread across days
    schedule = [
        Appointment(
            [patient0],
            DayHour(day=1, hour=9),
            treatment1,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
        Appointment(
            [patient0],
            DayHour(day=3, hour=9),
            treatment1,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
        Appointment(
            [patient0],
            DayHour(day=8, hour=12),
            treatment1,
            {rg_therapists: [therapist1], rg_rooms: [room1]},
        ),
        Appointment(
            [patient0],
            DayHour(day=7, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=8, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
        Appointment(
            [patient0],
            DayHour(day=9, hour=9),
            treatment0,
            {rg_therapists: [therapist0], rg_rooms: [room0]},
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Patient [0-9]* has 3 treatments for treatment .* scheduled between day 5 and 10",
    ):
        Solution(
            instance,
            schedule,
            patients_arrival,
        )
