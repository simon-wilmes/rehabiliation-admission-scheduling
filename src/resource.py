from src.time import Duration, DayHour
from src.types import RID, RGID


class ResourceGroup:
    def __init__(self, rgid: RGID, name: str = ""):
        self.rgid = rgid
        self.name = name

    def __str__(self):
        return f"RG{self.rgid}"


class Resource:

    def __init__(
        self,
        rid: RID,
        resource_group: list[ResourceGroup],
        name: str = "",
        unavailable_time_slots: list[tuple[DayHour, Duration]] = None,
    ):
        if unavailable_time_slots is None:
            unavailable_time_slots = []

        self.rid = rid
        self.resource_group = resource_group
        self.name = name
        self.unavailable_time_slots = unavailable_time_slots

    def __str__(self):
        return f"R{self.rid}"
