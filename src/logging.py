import logging
import colorlog
import sys


# Custom handler remains the same
class NormalStreamHandler(logging.StreamHandler):
    pass


# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Use sys.stdout directly without additional buffering layers
handler = NormalStreamHandler(sys.stdout)

# Set the colored formatter
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

# Add the handler to the logger
logger.addHandler(handler)

# Redirect `print` to logger's debug method for convenience
print = logger.debug

# Test the logging
print("This is a test debug message.")
