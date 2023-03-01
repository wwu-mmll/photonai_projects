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
         pipeline: PhotonaiPipeline = typer.Option(..., help="Specify PHOTONAI pipeline."),
         outer_cv: int = typer.Option(default=10, help="Number of outer folds."),
         inner_cv: int = typer.Option(default=10, help="Number of inner folds."),
         scaling: bool = typer.Option(True, help="Apply z-scaling of features?"),
         imputation: bool = typer.Option(True, help="Apply feature imputation?")):
    """
    Run PHOTONAI Analysis
    """
    print("Run PHOTONAI Analysis")
    print(f"Folder: {folder}")
    print(f"Task: {task}")
    print(f"Pipeline: {pipeline}")
    print(f"Outer CV: {outer_cv}")
    print(f"Inner CV: {inner_cv}")
    print(f"Scale features: {scaling}")
    print(f"Impute features: {imputation}")


if __name__ == "__main__":
    typer.run(main)
