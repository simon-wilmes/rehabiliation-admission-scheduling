from random import randint as rand
from random import random as randf
import numpy as np
from gurobipy import Model, GRB, quicksum


def set_max_k_values_to_one(matrix, k):
    # Flatten the 2D list and find the k-th largest value
    flattened = [val for row in matrix for val in row]
    threshold = sorted(flattened, reverse=True)[k - 1]

    # Create a new matrix with values set to 1 or 0
    result = [[1 if val >= threshold else 0 for val in row] for row in matrix]
    return result


n = 0


def run_algo(board, a, b, c):
    # OPTIMAL
    # All players' requirements
    players = ["A", "B", "C"]
    requirements = {"A": a, "B": b, "C": c}

    # Dimensions
    num_slots = len(board)
    num_resources = len(board[0])
    num_tasks = len(a)  # All players have the same number of tasks

    # Create the model
    model = Model("Player Task Scheduling")

    # Decision variables
    x = model.addVars(
        players,
        range(num_tasks),
        range(num_slots),
        range(num_resources),
        vtype=GRB.BINARY,
        name="x",
    )  # x[p, t, l, r] = 1 if player p completes task t at time l using resource r

    finish_time = model.addVars(players, vtype=GRB.INTEGER, name="finish_time")

    # Constraints

    # 1. Resource availability (only use resources available in the board)
    for l in range(num_slots):
        for r in range(num_resources):
            model.addConstr(
                quicksum(x[p, t, l, r] for p in players for t in range(num_tasks))
                <= board[l][r],
                name=f"resource_avail_t{l}_r{r}",
            )

    # 2. Each task is completed exactly once per player
    for p in players:
        for t in range(num_tasks):
            model.addConstr(
                quicksum(
                    x[p, t, l, r]
                    for l in range(num_slots)
                    for r in range(num_resources)
                )
                == 1,
                name=f"task_once_{p}_t{t}",
            )

    # 3. Sequential tasks for each player
    for p in players:
        for t in range(1, num_tasks):  # Skip t=0 since it has no prerequisite
            for l in range(num_slots):
                model.addConstr(
                    quicksum(x[p, t, lp, requirements[p][t]] for lp in range(l))
                    >= quicksum(
                        x[p, t - 1, l, requirements[p][t - 1]] for l in range(num_slots)
                    ),
                    name=f"sequential_{p}_t{t}_l{l}",
                )

    # 4. Each resource can only be used by one player per time slot
    for l in range(num_slots):
        for r in range(num_resources):
            model.addConstr(
                quicksum(x[p, t, l, r] for p in players for t in range(num_tasks)) <= 1,
                name=f"resource_unique_t{l}_r{r}",
            )

    # Define finish_time as the maximum time slot where the last task is completed
    for p in players:
        model.addConstr(
            finish_time[p]
            == quicksum(
                l * quicksum(x[p, num_tasks - 1, l, i] for i in range(num_resources))
                for l in range(num_slots)
            ),
            name=f"finish_time_{p}",
        )

    # Objective: Minimize average finish time
    model.setObjective(
        quicksum(finish_time[p] for p in players) / len(players), GRB.MINIMIZE
    )

    # Optimize
    model.optimize()
    # Print results
    if model.status == GRB.OPTIMAL:
        print("\nOptimal Solution Found:")
        for p in players:
            for t in range(num_tasks):
                for l in range(num_slots):
                    for r in range(num_resources):
                        if x[p, t, l, r].X > 0.5:
                            print(
                                f"Player {p} completes task {t} at time {l} using resource {r}"
                            )
        print("\nFinish Times:")
        for p in players:
            print(f"Player {p}: {finish_time[p].X}")
        print(f"\nAverage Finish Time: {model.objVal}")
    else:
        print("No optimal solution found.")


while True:
    a = [rand(0, 2) for _ in range(3)]
    b = [rand(0, 2) for _ in range(3)]
    c = [rand(0, 2) for _ in range(3)]

    l = rand(12, 21)
    board = []
    for i in range(l):
        board.append([])
        for _ in range(0, 3):
            board[-1].append(randf())
    max_value = rand(9, 12)
    board = set_max_k_values_to_one(board, max_value)
    run_algo(board, a, b, c)
