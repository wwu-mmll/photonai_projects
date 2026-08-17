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

### Sequential stopping

Permutation testing is usually the most expensive part of a project, and most
of that cost goes into confirming that analyses *without* signal have no
signal. Passing `sequential_metric` stops sampling as soon as
`max_exceedances` permutations have matched or beaten the observed value —
the point beyond which no number of further permutations could yield a small
p-value.

```python
project.run_permutation_test(
    name="second_feature_set",
    n_perms=1000,
    sequential_metric="explained_variance",
    stop_above_p=0.1,
)
```

`stop_above_p=0.1` reads as: *stop as soon as it is clear the p-value is at
least 0.1, and never before*. Anything that can still reach p < 0.1 runs to
the full budget at full resolution, down to `1 / (n_perms + 1)`. **Power is
unaffected** — only the analyses that were never going to be significant
finish early.

The threshold is the exceedance budget in disguise:

    max_exceedances = stop_above_p * n_perms

so `stop_above_p=0.1` with 1000 permutations is `max_exceedances=100`. Pass
whichever you prefer, but not both. Note that a *smaller* threshold stops
*later*: it takes more permutations to accumulate a larger budget.

| stop_above_p | permutations used when the true p is... |
|---|---|
| | `0.99` / `0.50` / `0.15` / `0.05` / `0.001` |
| 0.02 | 20 / 39 / 130 / 383 / 1000 |
| 0.10 | 101 / 200 / 665 / 1000 / 1000 |

The p-value follows Besag and Clifford (1991): `max_exceedances / L` if
sampling stopped at permutation `L`, and the usual
`(1 + exceedances) / (1 + n_perms)` otherwise. Both are valid p-values, so the
saving costs nothing in validity.

Note that the stopping rule watches one metric. When a run stops early,
`calculate_permutation_p_values()` reports a p-value for that metric only:
sampling stopped when *its* budget ran out, which says nothing about the
others. The decision is recorded in `sequential_permutation.json` inside the
analysis folder and can be inspected with `read_sequential_state()`.

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

Sequential stopping works here too:

```python
project.prepare_slurm_permutation_test(
    name="second_feature_set",
    n_perms=1000,
    conda_env="my_photonai_env",
    memory_per_cpu=2,
    n_jobs=20,
    run_time="0-02:00:00",
    random_state=1,
    sequential_metric="explained_variance",
    max_exceedances=20,
)
```

Every array task checks the exceedance budget before doing any work and exits
immediately if it is already spent.

**A single array saves little on its own.** If SLURM starts all tasks at once,
they all check before any results exist, find nothing, and each runs its full
share. The saving is proportional to how much of the array is still *queued*
when the budget runs out, which depends on partition contention rather than
on anything you control.

Use `prepare_staged_slurm_permutation_test` to make the saving reliable:

```python
project.prepare_staged_slurm_permutation_test(
    name="second_feature_set",
    n_perms=1000,
    conda_env="my_photonai_env",
    memory_per_cpu=2,
    n_jobs_per_stage=10,
    n_perms_per_stage=200,
    sequential_metric="explained_variance",
    stop_above_p=0.1,
    run_time="0-02:00:00",
)
```

This writes `slurm_stage.cmd` and `submit_stages.sh`. Submit with:

```bash
cd example_project/second_feature_set
./submit_stages.sh
```

Each stage is an array job depending on the previous one, so by the time a
later stage starts the earlier results are on disk and its tasks can see that
the budget is spent. Size a stage at roughly the number of permutations a null
analysis needs — about 200 for `stop_above_p=0.1` out of 1000.

Later stages are still *scheduled* even once sampling has stopped; they simply
exit within seconds. Queue slots are used, compute is not.

To inspect or drive the decision yourself:

```python
status = project.sequential_status(
    name="second_feature_set",
    metric="explained_variance",
    max_exceedances=100,
    n_perms=1000,
)
print(status["n_perms_used"], status["p_value"], status["should_continue"])
```

Each completed run also writes a small `permutation_summary.json` holding only
its mean outer-fold metrics. PHOTONAI's own results file runs to several
megabytes, so the stopping rule reads these summaries instead — which is what
makes checking after every permutation cheap. Runs from older versions are
summarised on first read.

## Next steps
See the Usage page for more details on:

- how to design your hyperpipe constructor, 
- how metrics and scorers are handled, 
- how to interpret the comparison reports.

See the API Reference for the full documentation of PhotonaiProject.
