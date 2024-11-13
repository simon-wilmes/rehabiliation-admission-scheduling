from src.time import Duration
from src.resource import Resource, ResourceGroup
from src.types import TID


class Treatment:

    def __init__(
        self,
        tid: TID,
        num_participants: int,
        duration: Duration,
        name: str = "",
        resources: dict[ResourceGroup, tuple[int, bool]] = None,
        # resources: dict[ResourceGroup, tuple[num_of_resource_group, resource_loyal?]] = None,
    ):
        if resources is None:
            resources = {}

        self.id = tid
        self.num_participants = num_participants
        self.duration = duration
        self.name = name
        self.resources = {rg: n for rg, (n, _) in resources.items()}
        self.loyalty = {rg: loyal for rg, (_, loyal) in resources.items()}

    def __str__(self):
        return f"T({self.name})"

    def __repr__(self):
        return self.__str__()
