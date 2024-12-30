def generate_unique_tuples(lists):
    def backtrack(current_tuple, used_elements, depth):
        # Base case: If the tuple is complete, yield it
        if depth == len(lists):
            yield tuple(current_tuple)
            return

        # Iterate through the current list at 'depth'
        for num in lists[depth]:
            if num not in used_elements:  # Check if num causes duplicates
                # Include num in the current tuple and mark it as used
                current_tuple.append(num)
                used_elements.add(num)

                # Recur to the next depth
                yield from backtrack(current_tuple, used_elements, depth + 1)

                # Backtrack: remove num and unmark it
                current_tuple.pop()
                used_elements.remove(num)

    # Start backtracking from depth 0
    return backtrack([], set(), 0)


from itertools import product


def product_of_unique_entries(lists):
    # Step 1: Generate all combinations (one entry from each list)
    all_combinations = product(*lists)

    # Step 2: Filter out combinations where integers are not unique
    result = [tup for tup in all_combinations if len(set(tup)) == len(tup)]

    return result


def generate_unique_tuples2(lists):
    # Sort lists by their length and keep track of original indices
    sorted_lists = sorted(enumerate(lists), key=lambda x: len(x[1]))
    indices, sorted_lists = zip(*sorted_lists)

    def backtrack(current_tuple, used_elements, depth):
        # Base case: If the tuple is complete, reorder it and yield
        if depth == len(sorted_lists):
            # Reorder tuple to match the original list order
            yield tuple(current_tuple[i] for i in indices)
            return

        # Iterate through the current list at 'depth'
        for num in sorted_lists[depth]:
            if num not in used_elements:  # Check for duplicates
                # Include num in the current tuple and mark it as used
                current_tuple.append(num)
                used_elements.add(num)

                # Recur to the next depth
                yield from backtrack(current_tuple, used_elements, depth + 1)

                # Backtrack: remove num and unmark it
                current_tuple.pop()
                used_elements.remove(num)

    # Start backtracking
    return backtrack([], set(), 0)


from collections import Counter


def generate_unique_tuples3(lists):
    # Step 1: Compute global frequency of all elements across all lists
    element_freq = Counter(num for sublist in lists for num in set(sublist))

    # Step 2: Sort lists based on their "restrictiveness"
    # A list is restrictive if it contains rare elements (low frequency)
    def list_restrictiveness(lst):
        return sum(element_freq[num] for num in set(lst)) / len(set(lst))

    sorted_lists_with_indices = sorted(
        enumerate(lists), key=lambda x: list_restrictiveness(x[1])
    )
    indices, sorted_lists = zip(*sorted_lists_with_indices)

    def backtrack(current_tuple, used_elements, depth):
        # Base case: If the tuple is complete, reorder and yield it
        if depth == len(sorted_lists):
            yield tuple(current_tuple[i] for i in indices)
            return

        # Step through the current list
        for num in sorted_lists[depth]:
            if num not in used_elements:
                # Add num to the tuple and mark as used
                current_tuple.append(num)
                used_elements.add(num)

                # Recursive call
                yield from backtrack(current_tuple, used_elements, depth + 1)

                # Backtrack
                current_tuple.pop()
                used_elements.remove(num)

    # Start backtracking
    return backtrack([], set(), 0)


# Example usage
lists = [[1, 2, 3, 2], [2, 3, 4], [1, 3, 5]]  # 4 elements  # 3 elements  # 3 elements

# Generate tuples
unique_tuples = list(generate_unique_tuples(lists))
print("Unique tuples:", unique_tuples)


from time import time

example = [
    [1, 2, 3, 4, 5, 6, 7, 8, 12],
    [2, 3, 4, 5, 6, 8, 12],
    # [3, 4, 5, 9, 12],
    # [1, 2, 12],
    # [1, 2, 3, 4],
    # [3, 4, 5, 6, 9],
    # [3, 4, 5, 6, 7],
    # [3, 5, 7, 8],
    # [1, 2, 3, 4, 5, 6],
    # [6, 7, 8, 9],
    # [10, 2, 3, 5],
    # [11, 2, 4],
]

timea = time()

resulta = list(generate_unique_tuples(example))
timea = time() - timea

timec = time()
resultb = list(generate_unique_tuples2(example))
timec = time() - timec

timeb = time()
resultc = list(generate_unique_tuples3(example))
timeb = time() - timeb
print(resulta)
print(resultb)
print(resultc)
assert set(resulta) == set(resultb)
assert set(resultb) == set(resultc)
assert set(resultc) == set(resulta)
print(timea, timec, timeb)
