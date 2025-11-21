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
    project.run_permutation_test(name=name, n_perms=1000, overwrite=True)

# ---------------------------------------------------------------------
# 4) Statistical comparison of analyses
# ---------------------------------------------------------------------
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
    method="permutation"
)

# Compare all pairs at once (optional)
multi_results = project.compare_multiple_analyses(
    analyses=["all_features", "first_feature_set", "second_feature_set"],
    method="permutation"
)

```