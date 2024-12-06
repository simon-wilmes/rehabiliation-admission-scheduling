from typing import TypeVar, Dict, Any
import os
from datetime import datetime

A = TypeVar("A")
B = TypeVar("B")


def slice_dict(d: Dict[A, B], conditions: tuple) -> dict:
    return {
        k: v
        for k, v in d.items()
        if all(
            cond is None or k[i] == cond for i, cond in enumerate(conditions)  # type: ignore
        )
    }


class MultiWriter:
    def __init__(self, streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_file_writer_context(
    solver_cls, instance, log_to_console: bool, log_to_file: bool
):
    import sys

    streams = []
    if log_to_file:  # type: ignore
        folder_path = "logs"
        os.makedirs(folder_path, exist_ok=True)
        instace_name = (
            instance.name.replace(" ", "_").replace("/", "_").split("data_")[-1]
        )
        file_stream = open(
            folder_path
            + f"/{solver_cls.__name__}_({instace_name})_{datetime.now().strftime("%m-%d-%H-%M")}.txt",
            "w+",
        )
        streams.append(file_stream)

    if log_to_console:  # type: ignore
        import sys

        console_stream = sys.stdout
        streams.append(console_stream)

    return MultiWriter(streams)
