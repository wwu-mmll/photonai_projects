import os
import sys
import shutil
import json
import importlib.util
import typer
from pathlib import Path

import numpy as np


class PhotonaiProject:
    def __init__(self,
                 project_folder: str,
                 feature_importances: bool = False):
        self.project_folder = project_folder
        self.feature_importances = feature_importances

        os.makedirs(self.project_folder, exist_ok=True)

    def run(self, name: str):
        # check that analysis folder exists
        if name not in os.listdir(self.project_folder):
            raise ValueError(f"Analysis {name} not found in project folder {self.project_folder}")

        analysis_folder = os.path.join(self.project_folder, name)
        data_folder = os.path.join(analysis_folder, 'data')

        pipe = self._load_hyperpipe(analysis_folder, name)
        pipe.output_settings.set_project_folder(analysis_folder)
        pipe.output_settings.set_log_file()
        pipe.name = name
        pipe.project_folder = analysis_folder

        # load data
        X = np.load(os.path.join(data_folder, 'X.npy'))
        y = np.load(os.path.join(data_folder, 'y.npy'))

        pipe.fit(X, y)

        # if you want to use feature_importances later, you can hook it here
        # if self.feature_importances:
        #     ...

        return pipe

    @staticmethod
    def _load_hyperpipe(analysis_folder: str, name: str, perm_run: bool = False):
        # ------------------------------------------------------------------
        # LOAD HYPERPIPE CONSTRUCTOR FROM HYPERPIPE SCRIPT
        # ------------------------------------------------------------------

        # 1) read metadata to get the constructor function name
        meta_path = os.path.join(analysis_folder, 'hyperpipe_meta.json')
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"No 'hyperpipe_meta.json' found for analysis '{name}' at {meta_path}. "
                f"Did you create this analysis with 'add_analysis'?"
            )

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        constructor_name = meta.get('name_hyperpipe_constructor', None)
        if constructor_name is None:
            raise KeyError(
                f"'name_hyperpipe_constructor' not found in {meta_path}"
            )

        # 2) load the hyperpipe_constructor.py as a module
        module_path = os.path.join(analysis_folder, 'hyperpipe_constructor.py')
        if not os.path.isfile(module_path):
            raise FileNotFoundError(
                f"No 'hyperpipe_constructor.py' found for analysis '{name}' at {module_path}"
            )

        spec = importlib.util.spec_from_file_location(
            f"hyperpipe_constructor_{name}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, constructor_name):
            raise AttributeError(
                f"Function '{constructor_name}' not found in {module_path}"
            )

        hyperpipe_constructor = getattr(module, constructor_name)

        # 3) build and run the Hyperpipe
        pipe = hyperpipe_constructor()  # adapt if your constructor needs arguments
        if perm_run:
            pipe.verbosity = -1
        return pipe

    def add_analysis(self, name, X, y,
                     hyperpipe_script: str,
                     name_hyperpipe_constructor: str,
                     **kwargs):
        """
        Create a new analysis folder, save X and y, copy the hyperpipe script,
        and store the constructor function name in a metadata file.
        """
        if hyperpipe_script is None:
            raise ValueError("hyperpipe_script must be provided in add_analysis.")
        if name_hyperpipe_constructor is None:
            raise ValueError("name_hyperpipe_constructor must be provided in add_analysis.")

        # create directories for analysis and data
        analysis_folder = os.path.join(self.project_folder, name)
        os.makedirs(analysis_folder, exist_ok=True)
        os.makedirs(os.path.join(analysis_folder, 'data'), exist_ok=True)

        # save data to numpy array
        np.save(os.path.join(analysis_folder, 'data', 'X.npy'), X)
        np.save(os.path.join(analysis_folder, 'data', 'y.npy'), y)

        # copy script that contains the hyperpipe definition
        shutil.copyfile(
            hyperpipe_script,
            os.path.join(analysis_folder, 'hyperpipe_constructor.py')
        )

        # save metadata (constructor function name etc.)
        meta = {
            'name_hyperpipe_constructor': name_hyperpipe_constructor
            # you could add more fields here (e.g. timestamp, description, etc.)
        }
        meta_path = os.path.join(analysis_folder, 'hyperpipe_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def list_analyses(self):
        analyses = [
            item for item in os.listdir(self.project_folder)
            if os.path.isdir(os.path.join(self.project_folder, item))
        ]
        print("Available PHOTONAI analyses are:")
        for analysis in analyses:
            print(f"  - {analysis}")

    def run_permutation_test(self, name: str, n_perms: int = 1000, random_state: int = 15, overwrite: bool = False):
        perm_runs = range(n_perms)
        self._run_permutation_test(name=name, random_state=random_state, n_perms=n_perms,
                                   overwrite=overwrite, perm_runs=perm_runs)

    def _run_permutation_test(self, name: str, random_state: int = 15, n_perms: int = 1000, overwrite: bool = False,
                              perm_runs: range = range(1000)):
        # check that analysis folder exists
        if name not in os.listdir(self.project_folder):
            raise ValueError(f"Analysis {name} not found in project folder {self.project_folder}")

        analysis_folder = os.path.join(self.project_folder, name)
        data_folder = os.path.join(analysis_folder, 'data')
        perm_folder = os.path.join(analysis_folder, 'permutations')

        # load data
        X = np.load(os.path.join(data_folder, 'X.npy'))
        y = np.load(os.path.join(data_folder, 'y.npy'))

        for perm_run in perm_runs:
            current_perm_folder = os.path.join(perm_folder, str(perm_run))
            if not overwrite and os.path.exists(os.path.join(current_perm_folder, 'photonai_results.json')):
                print(f"Skipping permutation {perm_run + 1}/{n_perms} as it already exists.")
                continue

            print(f"Running permutation {perm_run + 1}/{n_perms}")
            np.random.seed(random_state + perm_run)
            y_perm = np.random.permutation(y)
            pipe = self._load_hyperpipe(analysis_folder, name, perm_run=True)
            pipe.output_settings.set_project_folder(os.path.join(perm_folder, str(perm_run)))
            pipe.output_settings.set_log_file()
            pipe.name = name
            pipe.project_folder = os.path.join(perm_folder, str(perm_run))
            pipe.fit(X, y_perm)
            shutil.copyfile(
                os.path.join(pipe.output_settings.results_folder, 'photonai_results.json'),
                os.path.join(os.path.join(perm_folder, str(perm_run)), 'photonai_results.json')
            )
            shutil.rmtree(pipe.output_settings.results_folder)

    def run_permutation_test_slurm(self, name: str, n_perms: int = 1000, random_state: int = 15, overwrite: bool = False,
                                   slurm_job_id: int = None, n_perms_per_job: int = None):
        perms_to_do = np.arange((slurm_job_id - 1) * n_perms_per_job,
                                (slurm_job_id - 1) * n_perms_per_job + n_perms_per_job)
        self._run_permutation_test(name=name, random_state=random_state, n_perms=n_perms,
                                   overwrite=overwrite, perm_runs=perms_to_do)

    def prepare_slurm_permutation_test(self, name: str, n_perms: int,
                                       conda_env: str, memory_per_cpu: int, n_jobs: int, run_time: str = "0-01:00:00",
                                       random_state: int = 1
                                       ):
        if name not in os.listdir(self.project_folder):
            raise ValueError(f"Analysis {name} not found in project folder {self.project_folder}")

        analysis_folder = os.path.join(self.project_folder, name)
        # calculate the number of perms per job
        n_perms_per_job = int(n_perms / n_jobs)

        # copy script that contains the permutation test
        shutil.copyfile(os.path.abspath(__file__), os.path.join(self.project_folder, os.path.basename(__file__)))

        # create slurm script
        cmd = f"""#!/bin/bash

#SBATCH --job-name={name + "_perm_test"}
#SBATCH --output=logs/job_%a.log

#SBATCH --partition normal
#SBATCH --mem-per-cpu={memory_per_cpu}G
#SBATCH --time={run_time}
#SBATCH --array=1-{n_jobs}

# add python
module load palma/2021a
module load Miniconda3

# activate conda env
eval "$(conda shell.bash hook)"
conda activate {conda_env}


python ../project.py --project-folder ../../{self.project_folder} --analysis-name {name} --n-perms {n_perms} --slurm-job-id $SLURM_ARRAY_TASK_ID --n-perms-per-job {n_perms_per_job} --random-state {random_state}
"""
        with open(os.path.join(analysis_folder, "slurm_job.cmd"), "w") as text_file:
            text_file.write(cmd)

        return


def run_perm_job(project_folder: Path = typer.Option(
             ...,
             exists=True,
             file_okay=False,
             dir_okay=True,
             writable=True,
             readable=True,
             resolve_path=True,
             help="Path to perm folder which will contain all data and results used in the perm test."),
        analysis_name: str = typer.Option(..., help="name of the analysis"),
        n_perms: int = typer.Option(..., help="number of total permutation runs"),
        slurm_job_id: int = typer.Option(..., help="SLURM array task id"),
        n_perms_per_job: int = typer.Option(..., help="number of perms per SLURM job"),
        random_state: int = typer.Option(1, help="set random state")):

    project = PhotonaiProject(project_folder)
    project.run_permutation_test_slurm(name=analysis_name, n_perms=n_perms,
                                       slurm_job_id=slurm_job_id, n_perms_per_job=n_perms_per_job,
                                       random_state=random_state)


if __name__ == "__main__":
    typer.run(run_perm_job)