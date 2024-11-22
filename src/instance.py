import os
import ast  # To safely evaluate expressions from strings
from src.time import Duration, DayHour
from src.resource import Resource, ResourceGroup
from src.treatments import Treatment
from src.types import TID, RID, RGID, PID
from src.patients import Patient
from src.logging import logger, print
from typing import Any
import re


class Instance:
    def __init__(
        self,
        instance_data: dict[str, Any],
        resource_groups: dict[RGID, ResourceGroup],
        treatments: dict[TID, Treatment],
        resources: dict[RID, Resource],
        patients: dict[PID, Patient],
    ):
        self.beds_capacity = instance_data["num_beds"]
        conflict_groups: list[set[TID]] = instance_data.get("conflict_groups", [])
        # replace tid with treatment object
        self.conflict_groups = [
            set(treatments[tid] for tid in conflict_group)
            for conflict_group in conflict_groups
        ]

        self.workday_start: DayHour = (
            instance_data["workday_start"]
            if "workday_start" in instance_data
            else DayHour(hour=8, minutes=0)
        )
        self.workday_end = (
            instance_data["workday_end"]
            if "workday_end" in instance_data
            else DayHour(hour=17, minutes=0)
        )
        self.rolling_window_length: int = instance_data.get("rolling_window_length", 7)
        self.rolling_window_days: list[int] = instance_data.get(
            "rolling_windows_days", [0, 5, 10, 15, 20]
        )
        self.time_slot_length: Duration = instance_data.get(
            "time_slot_length", Duration(0, 15)
        )
        self.resource_groups: dict[RGID, ResourceGroup] = resource_groups
        self.treatments: dict[TID, Treatment] = treatments

        for rid, resource in resources.items():
            # make resources unavailable outside of work hours
            resource.unavailable_time_slots.extend(
                [
                    (DayHour(0, 0), DayHour(0, self.workday_start.hour), 1),
                    (DayHour(0, self.workday_end.hour), DayHour(0, 23, 59), 1),
                ]
            )

        self.resources: dict[RID, Resource] = resources
        self.patients: dict[PID, Patient] = patients

        self.week_length = 5

    def print_solution(self, solution):
        pass


def create_instance_from_file(file_path: str) -> "Instance":
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r") as file:
        lines = file.readlines()

    num_beds = None
    resource_groups = {}
    resources = {}
    treatments = {}
    patients = {}

    current_section = None
    headers = []
    instance_data = {}
    sections_ordering = [
        "INSTANCE",
        "RESOURCE_GROUPS",
        "RESOURCES",
        "TREATMENTS",
        "PATIENTS",
    ]
    for section in sections_ordering:
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines and comments
            if "#" in line:
                line = line.split("#")[0].strip()

            if line.startswith("[") and "]" in line:
                # Detect section headers
                start_index = line.find("[")
                end_index = line.find("]", start_index)
                section_line = line[start_index + 1 : end_index].strip()

                # Check if it's a DATA section
                if section_line.startswith("DATA:"):
                    # Extract section name
                    section_name = section_line[len("DATA:") :].strip()
                    current_section = section_name

                    # Now, after the closing ']', the rest of the line may contain headers
                    rest_of_line = line[end_index + 1 :].strip()
                    # Remove any inline comments
                    rest_of_line_no_comment = rest_of_line.split("#", 1)[0].strip()

                    # Check if there is a ':' indicating headers
                    if ":" in rest_of_line_no_comment:
                        colon_index = rest_of_line_no_comment.find(":")
                        headers_part = rest_of_line_no_comment[
                            colon_index + 1 :
                        ].strip()
                        headers = [h.strip() for h in headers_part.split(",")]
                    elif "#" in rest_of_line:
                        # Headers may be in the comment
                        comment_part = rest_of_line.split("#", 1)[1].strip()
                        headers = [h.strip() for h in comment_part.split(",")]
                    else:
                        headers = []
                else:
                    # Other sections like [INSTANCE]
                    current_section = section_line
                    headers = []
            elif current_section == "INSTANCE" and current_section == section:
                # Parse num_beds
                key, value = line.split(":", 1)
                value = value.strip()

                logger.debug(f"ADD {current_section} DATA")
                match key:
                    case "num_beds":
                        instance_data["num_beds"] = int(value)
                    case "workday_start":
                        instance_data["workday_start"] = DayHour(hour=float(value))
                    case "workday_end":
                        instance_data["workday_end"] = DayHour(hour=float(value))
                    case "rolling_window_length":
                        instance_data["rolling_window_length"] = int(value)
                    case "rolling_window_days":
                        instance_data["rolling_window_days"] = ast.literal_eval(value)
                    case "time_slot_length":
                        instance_data["time_slot_length"] = float(value)
                    case "conflict_groups":
                        instance_data["conflict_groups"] = ast.literal_eval(value)
                    case "day_start":
                        instance_data["day_start"] = int(value)
                    case _:
                        logger.warning(f"Unknown key {key} in INSTANCE section")

            elif current_section and headers:
                values = [v.strip() for v in line.split(";")]
                data = dict(zip(headers, values))
                parsed_data = {}

                if current_section == "RESOURCE_GROUPS" and current_section == section:
                    logger.debug(f"ADD {current_section}")
                    rgid = int(data["rgid"])
                    name = data["name"]
                    resource_groups[rgid] = ResourceGroup(rgid, name)
                elif current_section == "RESOURCES" and current_section == section:
                    logger.debug(f"ADD {current_section}")

                    for key in data:
                        value = data[key]
                        parsed_data[key] = parsing_parameter(value)

                    parsed_data["resource_group"] = resource_groups[int(data["rgid"])]

                    resource = Resource(**parsed_data)

                    resources[parsed_data["rid"]] = resource

                elif current_section == "TREATMENTS" and current_section == section:
                    logger.debug(f"ADD {current_section}")
                    parsed_data = {
                        key: parsing_parameter(value) for key, value in data.items()
                    }

                    resources_required = parsed_data["resources"]
                    assert type(resources_required) == dict
                    treatment_resources = {}
                    for rgid_key, (count, required) in resources_required.items():
                        rgid = int(rgid_key)
                        treatment_resources[resource_groups[rgid]] = (count, required)
                    parsed_data["resources"] = treatment_resources
                    treatment = Treatment(**parsed_data)  # type: ignore

                    treatments[parsed_data["tid"]] = treatment
                elif current_section == "PATIENTS" and current_section == section:
                    logger.debug(f"ADD {current_section}")
                    parsed_data = {
                        key: parsing_parameter(value) for key, value in data.items()
                    }

                    # Replace TID in already_resource_loyal
                    resource_loyal_dict = parsed_data.get("already_resource_loyal")
                    parsed_resource_loyal = {}
                    for (tid, rgid), rid_list in resource_loyal_dict.items():  # type: ignore
                        parsed_resource_loyal[
                            (treatments[tid], resource_groups[rgid])
                        ] = [resources[rid] for rid in rid_list]

                    parsed_data["already_resource_loyal"] = parsed_resource_loyal
                    # Replace TID in already_scheduled_treatments
                    already_scheduled_treatments: list[tuple[TID, int]] = (
                        parsed_data.get("already_scheduled_treatments", [])  # type: ignore
                    )
                    parsed_data["already_scheduled_treatments"] = [
                        (treatments[tid], count)
                        for tid, count in already_scheduled_treatments
                    ]

                    parsed_data["treatments"] = {
                        treatments[tid]: count
                        for tid, count in parsed_data[
                            "treatments"
                        ].items()  # type: ignore
                    }
                    patient = Patient(**parsed_data)  # type: ignore
                    patients[parsed_data["pid"]] = patient
                else:
                    # Not in the correct section
                    continue
            else:
                # Not in a known section or headers not defined
                continue

    return Instance(
        instance_data=instance_data,
        resource_groups=resource_groups,
        treatments=treatments,
        resources=resources,
        patients=patients,
    )


def parsing_parameter(value):
    try:
        parsed_value = eval(value)
        return parsed_value
    except (NameError, ValueError):
        return value
