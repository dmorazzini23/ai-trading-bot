import csv
import logging
import os

logger = logging.getLogger(__name__)


def log_metrics(record, filename="metrics/model_performance.csv"):
    """Append a metrics record to the CSV file."""

    # Protect file operations so metrics logging can't crash the bot
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_exists = os.path.isfile(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except Exception as e:  # pragma: no cover - best effort logging
        # Log exception and continue to avoid silent failure
        logger.warning(f"Failed to update metrics file {filename}: {e}")
