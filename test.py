from ortools.sat.python import cp_model

# Create a CP-SAT model
model = cp_model.CpModel()

# Define the start, size, and end of the interval
start = model.NewIntVar(0, 10, "start")
size = 5
end = model.NewIntVar(0, 10, "end")

# Define the presence literal
is_present = model.NewBoolVar("is_present")

# Create an optional interval variable
interval = model.NewOptionalIntervalVar(start, size, end, is_present, "interval")

# Retrieve the presence literal
presence_literal = interval.size_expr()
print(f"Presence literal: {presence_literal}")
