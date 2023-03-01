import typer
from enum import Enum
from pathlib import Path


class PhotonaiPipeline(str, Enum):
    basic = "basic"
    advanced = "advanced"
    ensemble = "ensemble"


class Task(str, Enum):
    regression = "regression"
    classification = "classification"


def main(folder: Path = typer.Option(
             ...,
             exists=False,
             file_okay=False,
             dir_okay=True,
             writable=True,
             readable=True,
             resolve_path=True,
             help="Path to project folder containing features and target."),
         task: Task = typer.Option(..., help="Specify type of the prediction task"),
         pipeline: PhotonaiPipeline = typer.Option(..., help="Specify PHOTONAI pipeline.")):
    """
    Run PHOTONAI Analysis
    """
    print("Run PHOTONAI Analysis")
    print(f"Folder: {folder}")
    print(f"Task: {task}")
    print(f"Pipeline: {pipeline}")


if __name__ == "__main__":
    typer.run(main)
