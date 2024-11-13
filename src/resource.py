from src.time import Duration, DayHour
from src.types import RID, RGID


class ResourceGroup:
    def __init__(self, rgid: RGID, name: str = ""):
        self.id = rgid
        self.name = name

    def __str__(self):
        return f"RG({self.name})"

    def __repr__(self):
        return self.__str__()


class Resource:

    def __init__(
        self,
        rid: RID,
        resource_group: ResourceGroup,
        name: str = "",
        unavailable_time_slots: list[tuple[DayHour, Duration]] = None,
        **kwargs,
    ):
        if unavailable_time_slots is None:
            unavailable_time_slots = []

        self.id = rid
        self.resource_group = resource_group
        self.name = name
        self.unavailable_time_slots = unavailable_time_slots

    def __str__(self):
        return f"R({self.resource_group.name}:{self.name})"

    def __repr__(self):
        return self.__str__()

    def is_available(self, d: DayHour):
        for slot in self.unavailable_time_slots:
            if slot[0] <= d <= slot[0] + slot[1]:
                return False
        return True
