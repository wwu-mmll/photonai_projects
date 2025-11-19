from pathlib import Path
from datetime import datetime


def find_latest_photonai_run(folder):
    folder = Path(folder)

    # Folders to skip
    exclude = {"permutations", "data", "logs", "tmp"}

    # Get all subfolders except excluded ones
    photonai_runs = [
        f for f in folder.iterdir()
        if f.is_dir() and f.name not in exclude
    ]

    # Parse datetime from folder names
    # Expecting folder names ending with: YYYY-MM-DD_HH-MM-SS
    runs_with_dates = []
    for f in photonai_runs:
        try:
            dt = datetime.strptime(f.name[-19:], "%Y-%m-%d_%H-%M-%S")
            runs_with_dates.append((dt, f))
        except ValueError:
            # If a folder does not match the pattern, skip it
            pass

    if not runs_with_dates:
        return None  # No valid run folders found

    # Pick the entry with the latest timestamp
    latest_folder = max(runs_with_dates, key=lambda x: x[0])[1]

    return str(latest_folder)