from src.time import Duration, DayHour
from src.types import RID, RGID
from copy import deepcopy
from itertools import product
from src.logging import logger


class ResourceGroup:
    def __init__(self, rgid: RGID, name: str = ""):
        self.id = rgid
        self.name = name

    def __str__(self):
        return f"RG({self.name})"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return self.id < other.id


class Resource:

    def __init__(
        self,
        rid: RID,
        resource_groups: list[ResourceGroup],
        name: str = "",
        unavailable_time_slots: list[tuple[DayHour, DayHour, int | None]] = list(),
        **kwargs,
    ):
        if unavailable_time_slots is None:
            unavailable_time_slots = []

        self.id = rid
        self.resource_groups = sorted(resource_groups)
        self.name = name
        self.unavailable_time_slots = unavailable_time_slots
        self._total_availability_hours_dict = {}

    def __str__(self):
        return f"R({self.name})"

    def __repr__(self):
        return self.__str__()

    def is_available(self, d: DayHour):
        for slot in self.unavailable_time_slots:
            if slot[2] is None:
                if slot[0] <= d < slot[1]:
                    return False
            else:
                if slot[0].day <= d.day:
                    new_date = d - slot[0]
                    new_end = slot[1] - slot[0]
                    new_date.day = new_date.day % slot[2]
                    if new_date < new_end:
                        return False

        return True

    def __lt__(self, other):
        return self.id < other.id

    def total_availability_hours(self, day: int) -> float:
        if day in self._total_availability_hours_dict:
            return self._total_availability_hours_dict[day]
        minutes_available = 0
        for hour, minute in product(range(24), range(60)):
            if self.is_available(DayHour(day, hour, minute)):
                minutes_available += 1

        self._total_availability_hours_dict[day] = minutes_available / 60
        return minutes_available / 60
