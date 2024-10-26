from src.treatments import Treatment
from src.time import DayHour, Duration
from src.resource import Resource, ResourceGroup


class Patient:
    def __init__(
        self,
        treatments: dict[Treatment, int],
        arrival_date: DayHour,
        already_admitted: bool = False,
        already_resource_loyal: dict[tuple[Treatment, ResourceGroup], Resource] = None,
        name: str = "",
    ):
        if already_resource_loyal is None:
            already_resource_loyal = {}
        self.arrival_date = arrival_date
        self.already_admitted = already_admitted
        self.already_resource_loyal = already_resource_loyal
        self.treatments = treatments
        self.name = name
