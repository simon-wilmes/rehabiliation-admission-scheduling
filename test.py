from ortools.sat.python import cp_model

# Create a CP-SAT model
model = cp_model.CpModel()


# Define the start, size, and end of the interval
start = model.new_int_var(0, 10, "start")
size = 5
end = model.new_int_var(0, 10, "end")
end2 = model.new_int_var(0, 20, "end2")


# define two intervals
interval1 = model.new_interval_var(start, size, end, "interval1")
interval2 = model.new_interval_var(end, 5, end2, "interval2")

# Add a no-overlap constraint
model.add_no_overlap([interval1, interval2])

solver = cp_model.CpSolver()
status = solver.Solve(model)
if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    print("start:", solver.Value(start))
    print("end:", solver.Value(end))
    print("end2:", solver.Value(end2))
else:
    print("No solution found.")
