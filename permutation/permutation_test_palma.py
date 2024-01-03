import os
import sys
import typer
import rich
import importlib
import shutil
import warnings

import numpy as np

from pathlib import Path

from pymodm import connect
from pymodm.errors import DoesNotExist

from photonai import PermutationTest
from photonai.processing.results_structure import MDBPermutationResults, MDBHyperpipe
from photonai.photonlogger.logger import logger


sys.stdout.flush()
# Temporarily suppress RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


class PermutationTestPalma(PermutationTest):
    def __init__(self, hyperpipe_constructor, permutation_id: str, slurm_job_id: int,
                 n_perms_per_job: int, n_perms=1000, random_state=15):
        super().__init__(hyperpipe_constructor=hyperpipe_constructor,
                         permutation_id=permutation_id,
                         n_perms=n_perms,
                         random_state=random_state,
                         n_processes=1,
                         verbosity=1)
        self.slurm_job_id = slurm_job_id
        self.n_perms_per_job = n_perms_per_job

    def fit(self, X, y, **kwargs):
        self.pipe = self.hyperpipe_constructor()

        # we need a mongodb to collect the results!
        if not self.pipe.output_settings.mongodb_connect_url:
            raise ValueError("MongoDB connection string must be given for permutation tests")

        # Get all specified metrics
        best_config_metric = self.pipe.optimization.best_config_metric
        self.metrics = PermutationTest.manage_metrics(self.pipe.optimization.metrics, self.pipe.elements[-1],
                                                      best_config_metric)

        # at first, we do a reference optimization
        y_true = y

        # Run with true labels
        connect(self.pipe.output_settings.mongodb_connect_url, alias="photon_core")
        # Check if it already exists in DB
        try:
            existing_reference = MDBHyperpipe.objects.raw({'permutation_id': self.mother_permutation_id,
                                                           'computation_completed': True}).first()
            if not existing_reference.permutation_test:
                existing_reference.permutation_test = MDBPermutationResults(n_perms=self.n_perms)
                existing_reference.save()
            # check if all outer folds exist
            logger.info(
                "Found hyperpipe computation with true targets, skipping the optimization process with true targets")
        except DoesNotExist:
            # if we haven't computed the reference value do it:
            logger.info("Calculating Reference Values with true targets.")
            try:
                self.pipe.permutation_id = self.mother_permutation_id
                self.pipe.fit(X, y_true, **kwargs)
                self.pipe.results.computation_completed = True
                self.pipe.results.permutation_test = MDBPermutationResults(n_perms=self.n_perms)
                self.clear_data_and_save(self.pipe)
                existing_reference = self.pipe.results

            except Exception as e:
                if self.pipe.results is not None:
                    self.pipe.results.permutation_failed = str(e)
                    logger.error(e)
                    PermutationTest.clear_data_and_save(self.pipe)
                raise e

        # find how many permutations have been computed already
        existing_permutations = list(MDBHyperpipe.objects.raw({'permutation_id': self.permutation_id,
                                                               'computation_completed': True}).only('name'))
        existing_permutations = [int(perm_run.name.split('_')[-1]) for perm_run in existing_permutations]

        # calculate which perm jobs need to be computed
        perms_to_do = np.arange(self.slurm_job_id * self.n_perms_per_job,
                                self.slurm_job_id * self.n_perms_per_job + self.n_perms_per_job)

        # check which have already been computed
        if len(existing_permutations) > 0:
            perms_to_do = set(perms_to_do) - set(existing_permutations)

        print(f"{self.n_perms_per_job - len(perms_to_do)} of {self.n_perms_per_job} permutation runs done")

        if len(perms_to_do) > 0:
            # create permutation labels
            np.random.seed(self.random_state)
            self.permutations = [np.random.permutation(y_true) for _ in range(self.n_perms)]

            for perm_run in perms_to_do:
                PermutationTest.run_parallelized_permutation(self.hyperpipe_constructor, X, perm_run,
                                                             self.permutations[perm_run],
                                                             self.permutation_id, self.verbosity, **kwargs)

        return self


class PermutationTestSetup:
    def __init__(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, hyperpipe_script: str, name_hyperpipe_constructor: str,
                 perm_id: str, perm_folder: str,
                 conda_env: str, memory_per_cpu: int, n_jobs: int, run_time: str = "0-01:00:00", n_perms: int = 1000,
                 random_state: int = 1):
        self.X = X
        self.y = y
        self.groups = groups
        self.hyperpipe_script = hyperpipe_script
        self.name_hyperpipe_constructor = name_hyperpipe_constructor
        self.perm_id = perm_id
        self.perm_folder = perm_folder
        self.conda_env = conda_env
        self.run_time = run_time
        self.memory_per_cpu = memory_per_cpu
        self.n_jobs = n_jobs
        self.n_perms = n_perms
        self.random_state = random_state

        self.data_folder = os.path.join(self.perm_folder, 'data')
        self.log_folder = os.path.join(self.perm_folder, 'logs')
        self.results_folder = os.path.join(self.perm_folder, 'results')

    def prepare(self):
        # create directories for perm data and results
        os.makedirs(self.perm_folder, exist_ok=True)
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)

        # save data to numpy array
        np.save(os.path.join(self.data_folder, 'X'), self.X)
        np.save(os.path.join(self.data_folder, 'y'), self.y)
        if self.groups is not None:
            np.save(os.path.join(self.data_folder, 'groups'), self.groups)

        # calculate the number of perms per job
        n_perms_per_job = int(self.n_perms / self.n_jobs)

        # copy script that contains the hyperpipe definition
        shutil.copyfile(self.hyperpipe_script, os.path.join(self.perm_folder, 'hyperpipe_constructor.py'))

        # copy script that contains the permutation test
        shutil.copyfile(os.path.abspath(__file__), os.path.join(self.perm_folder, os.path.basename(__file__)))

        # create slurm script
        cmd = f"""#!/bin/bash

#SBATCH --job-name={self.perm_id}
#SBATCH --output=logs/job_%a.log

#SBATCH --partition normal
#SBATCH --mem-per-cpu={self.memory_per_cpu}G
#SBATCH --time={self.run_time}
#SBATCH --array=1-{self.n_jobs}

# add python
module load palma/2021a
module load Miniconda3

# activate conda env
eval "$(conda shell.bash hook)"
conda activate {self.conda_env}

python permutation_test_palma.py --perm-folder . --name-hyperpipe-constructor {self.name_hyperpipe_constructor} --perm-id {self.perm_id} --n-perms {self.n_perms} --slurm-job-id $SLURM_ARRAY_TASK_ID --n-perms-per-job {n_perms_per_job} --random-state {self.random_state}
"""
        with open(os.path.join(self.perm_folder, "slurm_job.cmd"), "w") as text_file:
            text_file.write(cmd)


def run_perm_job(perm_folder: Path = typer.Option(
             ...,
             exists=True,
             file_okay=False,
             dir_okay=True,
             writable=True,
             readable=True,
             resolve_path=True,
             help="Path to perm folder which will contain all data and results used in the perm test."),
        n_perms: int = typer.Option(..., help="number of total permutation runs"),
        name_hyperpipe_constructor: str = typer.Option(default='create_hyperpipe',
                                                  help="name of the function that creates the hyperpipe"),
        perm_id: str = typer.Option(..., help="permutation id"),
        slurm_job_id: int = typer.Option(..., help="SLURM array task id"),
        n_perms_per_job: int = typer.Option(..., help="number of perms per SLURM job"),
        random_state: int = typer.Option(1, help="set random state")):

    # import function that will create hyperpipe
    sys.path.append(os.path.abspath(perm_folder))
    imported_module = importlib.import_module('hyperpipe_constructor')
    hyperpipe_constructor = getattr(imported_module, name_hyperpipe_constructor)

    # run perm jobs
    perm_tester = PermutationTestPalma(hyperpipe_constructor, n_perms=n_perms, random_state=random_state,
                                       permutation_id=perm_id, slurm_job_id=slurm_job_id,
                                       n_perms_per_job=n_perms_per_job)

    # load data
    X = np.load(os.path.join(perm_folder, 'data', 'X.npy'))
    y = np.load(os.path.join(perm_folder, 'data', 'y.npy'))
    if os.path.exists(os.path.join(perm_folder, 'data', 'groups.npy')):
        groups = np.load(os.path.join(perm_folder, 'data', 'groups.npy'))
        perm_tester.fit(X, y, groups=groups)
    else:
        perm_tester.fit(X, y)


if __name__ == "__main__":
    typer.run(run_perm_job)
