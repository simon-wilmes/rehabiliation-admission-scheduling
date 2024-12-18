import logging
import colorlog
import io
import sys

# Create a buffered stdout stream
buffered_stdout = io.TextIOWrapper(
    open(sys.stdout.fileno(), mode="wb", buffering=2048),  # Large buffer
    write_through=True,  # Do not flush immediately
)
DISABLE_FLUSH = False


# Custom handler to suppress automatic flush
class BufferedStreamHandler(logging.StreamHandler):
    def flush(self):
        """Override flush to prevent automatic flushing."""
        pass


class NormalStreamHandler(logging.StreamHandler):
    pass


# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not DISABLE_FLUSH:
    handler = NormalStreamHandler(buffered_stdout)
else:
    # Create a custom buffered handler
    handler = BufferedStreamHandler(buffered_stdout)

# Set up color logging with colorlog
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
)

# Add handler to logger
logger.addHandler(handler)

# Redirect `print` to logger
print = logger.debug
