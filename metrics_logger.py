import csv
import os
import logging


def log_metrics(record, filename="metrics/model_performance.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except OSError as exc:
        logging.getLogger(__name__).error("Failed to log metrics: %s", exc)
