# Getting started

## Installation

Install the package (and **PHOTONAI**) into your environment:

```bash
pip install photonai photonai-projects 
```

## Basic concepts
A **PhotonaiProject** manages multiple **PHOTONAI** analyses in a single
project folder. Each analysis has its own subfolder containing:

- a hyperpipe constructor script (hyperpipe_constructor.py)

- a metadata file specifying the function name that creates the hyperpipe (hyperpipe_meta.json)

- a data/ folder with X.npy and y.npy

- (optionally) a permutations/ folder for results of the permutation test

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
```

You can now add multiple analyses that e.g. use different sets of features. You need to pass the data (X, y) as if
you were to call .fit() on a hyperpipe. The data arrays are then saved to disk which makes it easy to access them
when running an analysis or performing the permutation test. This also makes it easy to simply rsync everything
to an HPC cluster and run analyses there. Instead of creating a hyperpipe during runtime, you pass the location
of a Python script that contains a function (hyperpipe constructor) that creates the PHOTONAI Hyperpipe you want
to use in this project.
```python
# ---------------------------------------------------------------------
# 1) Register analyses
# ---------------------------------------------------------------------
for name, current_X in [("all_features", X), ("first_feature_set", X_1), ("second_feature_set", X_2)]:
    project.add(
        name=name,
        X=current_X,
        y=y,
        hyperpipe_script="path/to/hyperpipe_constructor.py",
        name_hyperpipe_constructor="create_hyperpipe",
    )

project.list_analyses()
```
Your PHOTONAI Hyperpipe constructor might look something like this.
```python
from photonai import Hyperpipe, PipelineElement
from sklearn.model_selection import KFold


def create_hyperpipe():
    my_pipe = Hyperpipe('',
                        optimizer='grid_search',
                        metrics=['accuracy', 'precision', 'recall'],
                        best_config_metric='accuracy',
                        outer_cv=KFold(n_splits=10),
                        inner_cv=KFold(n_splits=2),
                        verbosity=1,
                        project_folder='')

    # Add transformer elements
    my_pipe += PipelineElement("StandardScaler", hyperparameters={},
                               test_disabled=True, with_mean=True, with_std=True)

    my_pipe += PipelineElement("PCA", test_disabled=False)

    # Add estimator
    my_pipe += PipelineElement("SVC", hyperparameters={'kernel': ['linear', 'rbf']},
                               gamma='scale', max_iter=10000)

    return my_pipe
```

Now, when you want to run an analysis, you can refer to it by its name and simply call .run(). You can
perform a permutation test in the same way.
```python
# ---------------------------------------------------------------------
# 2) Run analyses
# ---------------------------------------------------------------------
project.run(name="all_features")
project.run(name="first_feature_set")
project.run(name="second_feature_set")

# ---------------------------------------------------------------------
# 3) Run permutation tests (local example)
# ---------------------------------------------------------------------
# Use a small number of permutations for testing; increase for real studies.
project.run_permutation_test(name="all_features", n_perms=1000)
project.run_permutation_test(name="first_feature_set", n_perms=1000)
project.run_permutation_test(name="second_feature_set", n_perms=1000)
```

If you want to compare two PHOTONAI analyses, you can use the .compare_analyses() method which either uses
the Nadeau-Bengio corrected t-test or relies on the permutations that have been computed in the individual 
significance test of each analysis.
```python
# ---------------------------------------------------------------------
# 4) Statistical comparison of analyses
# ---------------------------------------------------------------------
# For the Nadeau–Bengio test you must provide n_train and n_test as used
# during cross-validation. Here we give a simple example.
# Compare two analyses (Nadeau–Bengio corrected t-test)
project.compare_analyses(
    first_analysis="first_feature_set",
    second_analysis="second_feature_set",
    method="nadeau-bengio",
    n_train=9,
    n_test=1,
)

# Compare two analyses (permutation-based)
project.compare_analyses(
    first_analysis="all_features",
    second_analysis="second_feature_set",
    method="permutation",
    n_perms=1000,
)

# Compare all pairs at once (optional)
multi_results = project.compare_multiple_analyses(
    analyses=["all_features", "first_feature_set", "second_feature_set"],
    method="permutation",
    n_perms=1000,
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
