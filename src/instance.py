import os
import ast  # To safely evaluate expressions from strings
from src.time import Duration, DayHour
from src.resource import Resource, ResourceGroup
from src.treatments import Treatment
from src.types import TID, RID, RGID, PID
from src.patients import Patient


cls_mapper = {
    "RESOURCE_GROUPS": ResourceGroup,
    "RESOURCES": Resource,
    "TREATMENTS": Treatment,
    "PATIENTS": Patient,
}


class Instance:
    def __init__(
        self,
        num_beds: int,
        resource_groups: dict[RGID, ResourceGroup],
        treatments: dict[TID, Treatment],
        resources: dict[RID, Resource],
        patients: dict[PID, Patient],
    ):
        self.num_beds = num_beds
        self.resource_groups = resource_groups
        self.treatments = treatments
        self.resources = resources
        self.patients = patients


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
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # Skip empty lines and comments

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
                    headers_part = rest_of_line_no_comment[colon_index + 1 :].strip()
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
        elif current_section == "INSTANCE":
            # Parse num_beds
            if line.startswith("num_beds:"):
                num_beds = int(line.split(":", 1)[1].strip())
        elif current_section and headers:
            values = [v.strip() for v in line.split(";")]
            data = dict(zip(headers, values))

            if current_section == "RESOURCE_GROUPS":
                rgid = int(data["RGID"])
                name = data["NAME"]
                resource_groups[rgid] = ResourceGroup(rgid, name)
            elif current_section == "RESOURCES":
                rid = int(data["RID"])
                rgid = int(data["RGID"])
                name = data["NAME"]
                unavailable_time_slots = data["UNAVAILABLE_TIME_SLOTS"]
                if unavailable_time_slots == "None":
                    unavailable_time_slots = []
                else:
                    unavailable_time_slots = ast.literal_eval(unavailable_time_slots)
                resource = Resource(
                    rid, resource_groups[rgid], name, unavailable_time_slots
                )
                resources[rid] = resource
            elif current_section == "TREATMENTS":
                tid = int(data["TID"])
                num_participants = int(data["NUM_PARTICIPANTS"])
                name = data["NAME"]
                duration = Duration.from_string(data["DURATION"])
                resources_required = ast.literal_eval(data["RESOURCES"])
                treatment_resources = {}
                for rgid_key, (count, required) in resources_required.items():
                    rgid = int(rgid_key)
                    treatment_resources[resource_groups[rgid]] = (count, required)
                treatment = Treatment(
                    tid, num_participants, name, duration, treatment_resources
                )
                treatments[tid] = treatment
            elif current_section == "PATIENTS":
                pid = int(data["PID"])
                name = data["NAME"]
                patient_treatments = ast.literal_eval(data["TREATMENTS"])
                treatments_required = {
                    treatments[int(tid)]: count
                    for tid, count in patient_treatments.items()
                }
                arrival_date = DayHour.from_string(data["ARRIVAL_DATE"])
                already_admitted = bool(int(data["ALREADY_ADMITTED"]))
                resource_loyal_str = data.get("ALREADY_RESOURCE_LOYAL", "None")
                if resource_loyal_str == "None":
                    resource_loyal = {}
                else:
                    resource_loyal_dict = ast.literal_eval(resource_loyal_str)
                    parsed_resource_loyal = {}
                    for (tid_str, rgid_str), rid in resource_loyal_dict.items():
                        tid = int(tid_str)
                        rgid = int(rgid_str)
                        rid = int(rid)
                        parsed_resource_loyal[
                            (treatments[tid], resource_groups[rgid])
                        ] = resources[rid]
                    resource_loyal = parsed_resource_loyal
                patient = Patient(
                    treatments_required,
                    arrival_date,
                    already_admitted,
                    resource_loyal,
                    name,
                )
                patients[pid] = patient
        else:
            # Not in a known section or headers not defined
            continue

    if num_beds is None:
        raise ValueError("num_beds not specified in the instance file.")

    return Instance(
        num_beds=num_beds,
        resource_groups=resource_groups,
        treatments=treatments,
        resources=resources,
        patients=patients,
    )
