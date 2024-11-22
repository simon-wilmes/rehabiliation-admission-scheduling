from src.treatments import Treatment
from src.time import DayHour, Duration
from src.resource import Resource, ResourceGroup
from collections import defaultdict


class Patient:

    def __init__(
        self,
        pid: int,
        treatments: dict[Treatment, int],
        length_of_stay: int,
        earliest_admission_date: DayHour,
        admitted_before_date: DayHour,
        already_admitted: bool = False,
        already_resource_loyal: dict[
            tuple[Treatment, ResourceGroup], list[Resource]
        ] = dict(),
        already_scheduled_treatments: list[tuple[Treatment, int]] = list(),
        name: str = "",
        **kwargs,
    ):
        if already_resource_loyal is None:
            already_resource_loyal = {}
        self.id = pid

        self.length_of_stay = length_of_stay
        self.earliest_admission_date = earliest_admission_date
        self.admitted_before_date = admitted_before_date
        self.already_admitted = already_admitted
        self.already_resource_loyal = already_resource_loyal
        self.already_scheduled_treatments = defaultdict(
            int, {m: v for m, v in already_scheduled_treatments}
        )
        self.treatments = treatments
        self.name = name

    def __str__(self):
        return f"P({self.name})"

    def __repr__(self):
        return self.__str__()
