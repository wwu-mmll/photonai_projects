# Getting started

This page shows how to set up a simple PHOTONAI project using
`PhotonaiProject`, run analyses, perform permutation tests, and
statistically compare different analyses.

## Installation

Install the package (and PHOTONAI) into your environment:

```bash
pip install photonai photonai-projects 
```

## Basic concepts
A **PhotonaiProject** manages multiple PHOTONAI analyses in a single
project folder. Each analysis has its own subfolder containing:

- a hyperpipe constructor script (hyperpipe_constructor.py)

- a metadata file (hyperpipe_meta.json)

- a data/ folder with X.npy and y.npy

- (optionally) a permutations/ folder for permutation tests

The typical workflow is:

1. Create a project with PhotonaiProject. 
2. Add analyses (data + hyperpipe constructor). 
3. Run analyses to train and evaluate the models. 
4. Run permutation tests to obtain null distributions. 
5. Compare analyses statistically.

## Minimal example
Below is a complete example using the breast cancer dataset from
scikit-learn. We create three analyses using different feature sets,
run them, run permutation tests, and then compare them statistically.


```python
from photonai_projects.project import PhotonaiProject
from sklearn.datasets import load_breast_cancer

# Load example data
X, y = load_breast_cancer(return_X_y=True)

# Split features into different sets
X_1 = X[:, :3]
X_2 = X[:, 3:6]

# Create a project
project = PhotonaiProject(project_folder="example_project")

# ---------------------------------------------------------------------
# 1) Register analyses
# ---------------------------------------------------------------------
for name, current_X in [
    ("all_features", X),
    ("first_feature_set", X_1),
    ("second_feature_set", X_2),
]:
    project.add(
        name=name,
        X=current_X,
        y=y,
        hyperpipe_script="path/to/hyperpipe_constructor.py",
        name_hyperpipe_constructor="create_hyperpipe",
    )

project.list_analyses()

# ---------------------------------------------------------------------
# 2) Run analyses
# ---------------------------------------------------------------------
for name in ["all_features", "first_feature_set", "second_feature_set"]:
    project.run(name=name)

# ---------------------------------------------------------------------
# 3) Run permutation tests (local example)
# ---------------------------------------------------------------------
# Use a small number of permutations for testing; increase for real studies.
for name in ["all_features", "first_feature_set", "second_feature_set"]:
    project.run_permutation_test(name=name, n_perms=10, overwrite=True)

# ---------------------------------------------------------------------
# 4) Statistical comparison of analyses
# ---------------------------------------------------------------------
# For the Nadeau–Bengio test you must provide n_train and n_test as used
# during cross-validation. Here we give a simple example.
n_samples = X.shape[0]
n_train = int(0.8 * n_samples)
n_test = n_samples - n_train

# Compare two analyses (Nadeau–Bengio corrected t-test)
project.compare_analyses(
    first_analysis="first_feature_set",
    second_analysis="second_feature_set",
    method="nadeau-bengio",
    n_train=n_train,
    n_test=n_test,
)

# Compare two analyses (permutation-based)
project.compare_analyses(
    first_analysis="all_features",
    second_analysis="second_feature_set",
    method="permutation",
    n_perms=10,
)

# Compare all pairs at once (optional)
multi_results = project.compare_multiple_analyses(
    analyses=["all_features", "first_feature_set", "second_feature_set"],
    method="permutation",
    n_perms=10,
)
print(multi_results.head())
```

## Running permutation tests on a SLURM cluster
For large numbers of permutations, you can distribute them across a
SLURM array:

```python
project.prepare_slurm_permutation_test(
    name="second_feature_set",
    n_perms=1000,
    conda_env="my_photonai_env",
    memory_per_cpu=2,
    n_jobs=20,
    run_time="0-02:00:00",
    random_state=1,
)
```

This creates a slurm_job.cmd script in the analysis folder which you
can submit with:

```bash
cd example_project/second_feature_set
sbatch slurm_job.cmd
```

Each array job will call the Typer CLI entry point run_perm_job and
execute a subset of permutation runs.

## Next steps
See the Usage page for more details on:

- how to design your hyperpipe constructor, 
- how metrics and scorers are handled, 
- how to interpret the comparison reports.

See the API Reference for the full documentation of PhotonaiProject.
