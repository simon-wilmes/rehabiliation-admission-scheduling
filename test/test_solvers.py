import pytest
from src.solvers.solvers import Solver
from src.solvers.mip import MIPSolver
from src.solvers.cp_or import CPSolver
from src.instance import Instance
from src.patients import Patient
from src.treatments import Treatment
from src.resource import Resource, ResourceGroup
from src.time import DayHour, Duration
from src.solution import NO_SOLUTION_FOUND
from typing import Type

@pytest.fixture
def setup_instance():
    # Create resources
    rg_therapists = ResourceGroup(0, "therapists")
    therapist0 = Resource(
        0,
        rg_therapists,
        "therapist0",
        unavailable_time_slots=[(DayHour(0, 9), DayHour(0, 18), None)],
    )
    therapist1 = Resource(
        1,
        rg_therapists,
        "therapist1",
        unavailable_time_slots=[(DayHour(0, 9), DayHour(0, 18), None)],
    )
    therapist2 = Resource(
        2,
        rg_therapists,
        "therapist2",
        unavailable_time_slots=[(DayHour(0, 9), DayHour(0, 18), None)],
    )

    # Create treatments
    treatment0 = Treatment(
        tid=0,
        num_participants=2,
        duration=Duration(hours=1),
        name="therapy",
        resources={
            rg_therapists: (2, False),
        },
    )

    # Create patients
    patient0 = Patient(
        pid=0,
        treatments={treatment0: 1},
        length_of_stay=1,
        earliest_admission_date=DayHour(day=0, hour=0),
        admitted_before_date=DayHour(day=1, hour=0),
        already_admitted=False,
        already_resource_loyal={},
        name="patient0",
    )
    patient1 = Patient(
        pid=1,
        treatments={treatment0: 1},
        length_of_stay=1,
        earliest_admission_date=DayHour(day=0, hour=0),
        admitted_before_date=DayHour(day=1, hour=0),
        already_admitted=False,
        already_resource_loyal={},
        name="patient1",
    )
    patient2 = Patient(
        pid=2,
        treatments={treatment0: 1},
        length_of_stay=1,
        earliest_admission_date=DayHour(day=0, hour=0),
        admitted_before_date=DayHour(day=1, hour=0),
        already_admitted=False,
        already_resource_loyal={},
        name="patient2",
    )

    instance_data = {
        "num_beds": 3,
        "workday_start": DayHour(hour=8),
        "workday_end": DayHour(hour=9),
        "rolling_window_length": 7,
        "rolling_windows_days": [0, 5, 10],
        "conflict_groups": [],
        "time_slot_length": Duration(hours=0, minutes=15),
    }
    return (
        instance_data,
        rg_therapists,
        therapist0,
        therapist1,
        therapist2,
        treatment0,
        patient0,
        patient1,
        patient2,
    )


@pytest.mark.parametrize("solver", [MIPSolver, CPSolver])
def test_infeasible_solution(solver: Type[Solver], setup_instance):
    # Create an instance with a patient that requires a treatment that is not available
    # on the day the patient is admitted
    (
        instance_data,
        rg_therapists,
        therapist0,
        therapist1,
        therapist2,
        treatment0,
        patient0,
        patient1,
        patient2,
    ) = setup_instance

    instance = Instance(
        instance_data=instance_data,
        resource_groups={0: rg_therapists},
        treatments={0: treatment0},
        resources={0: therapist0, 1: therapist1, 2: therapist2},
        patients={0: patient0, 1: patient1, 2: patient2},
    )

    solver_instance = solver(instance)
    solver_instance.create_model()
    solution = solver_instance.solve_model()
    assert solution is NO_SOLUTION_FOUND
