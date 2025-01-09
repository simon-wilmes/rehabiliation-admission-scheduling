from collections import Counter

with open(
    "output/special_files/LBBDSolver_1_c3612d5807c7602ce37adc93d991fdf2_53000705.out",
    "r",
) as f:
    file_content = f.readlines()

import re

regex_forbidden_vars = "forbidden_vars: (.*)"
forbidden_vars = re.findall(regex_forbidden_vars, "\\n".join(file_content))

eval_forbidden_vars = [eval(f[:-4]) for f in forbidden_vars]

from itertools import product
from collections import defaultdict

smallest_partial_mapping = defaultdict(list)
i = 0
for prev_f, next_f in product(eval_forbidden_vars, eval_forbidden_vars):
    i += 1
    if i % 100 == 0:
        print(i)
        print(i / len(eval_forbidden_vars) ** 2)
    if prev_f == next_f:
        continue

    # Count occurrences of each dictionary in both lists
    start_counter = Counter(map(lambda x: tuple(sorted(x.items())), prev_f))
    end_counter = Counter(map(lambda x: tuple(sorted(x.items())), next_f))

    # Find the overlap (common elements)
    common_counter = start_counter & end_counter

    # Calculate elements to remove and add
    removed_elements = list((start_counter - common_counter).elements())
    added_elements = list((end_counter - common_counter).elements())

    # Convert frozensets back to dicts for readability
    removed_elements = [dict(e) for e in removed_elements]
    added_elements = [dict(e) for e in added_elements]

    # Output results
    print(f"Largest partial mapping size: {sum(common_counter.values())}")
    print(f"Elements to remove from start: {len(removed_elements)}")
    print(f"Removed elements: {removed_elements}")
    print(f"Elements to add to start: {len(added_elements)}")
    print(f"Added elements: {added_elements}")

    smallest_partial_mapping[sum(common_counter.values())].append(
        (removed_elements, added_elements)
    )

print(smallest_partial_mapping)
a = 0
for key in sorted(smallest_partial_mapping.keys()):
    print(key, len(smallest_partial_mapping[key]) / len(eval_forbidden_vars) ** 2)
    a += len(smallest_partial_mapping[key]) / len(eval_forbidden_vars) ** 2
print(a)
print(len(smallest_partial_mapping) ** 2)
input()
