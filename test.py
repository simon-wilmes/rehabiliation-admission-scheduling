import inspect



def some_function():
    print("Starting function")
    raise ValueError("This is a ValueError")
    print("This line would normally not execute")
    raise RuntimeError("This is a RuntimeError")
    print("End of function")


def execute_ignoring_exceptions(func):
    # Get the source code of the function
    func_code = inspect.getsource(func)
    # Create an exec-safe version of the code
    exec_code = f"""
try:
    {func_code}
except Exception as e:
    print(f"Ignored exception: {{e}}")
"""
    # Execute the code in a clean environment
    exec(exec_code)


# Usage
execute_ignoring_exceptions(some_function)
