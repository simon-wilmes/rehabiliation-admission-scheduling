from src.time import Duration
from src.resource import Resource, ResourceGroup
from src.types import TID


class Treatment:

    def __init__(
        self,
        tid: TID,
        max_participants: int,
        duration: Duration,
        name: str = "",
        resources: dict[ResourceGroup, int] = dict(),
        min_participants: int = 0,
        rest_time: Duration = Duration(hours=0),
        # resources: dict[ResourceGroup, tuple[num_of_resource_group, resource_loyal?]] = None,
    ):
        if resources is None:
            resources = {}

        self.id = tid
        self.max_num_participants = max_participants
        self.min_num_participants = min_participants
        self.duration = duration
        self.name = name
        self.resources = {rg: n for rg, n in resources.items()}
        self.rest_time: Duration = rest_time

    def __str__(self):
        return f"T({self.name})"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return self.id < other.id
