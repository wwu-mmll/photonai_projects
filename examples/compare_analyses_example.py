from photonai_projects.project import PhotonaiProject
from sklearn.datasets import load_breast_cancer

# Load example data
X, y = load_breast_cancer(return_X_y=True)

X_1 = X[:, :3]
X_2 = X[:, 3:6]

project = PhotonaiProject(project_folder="example_project")

# ---------------------------------------------------------------------
# 1) Create analyses
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
        hyperpipe_script="hyperpipe_constructor.py",
        name_hyperpipe_constructor="create_hyperpipe",
    )

project.list_analyses()

# ---------------------------------------------------------------------
# 2) Run all analyses
# ---------------------------------------------------------------------
for name in ["all_features", "first_feature_set", "second_feature_set"]:
    project.run(name=name)

# ---------------------------------------------------------------------
# 3) Run permutation tests locally (example: fewer perms for speed)
# ---------------------------------------------------------------------
for name in ["all_features", "first_feature_set", "second_feature_set"]:
    project.run_permutation_test(name=name, n_perms=10, overwrite=True)

# (Optional) Prepare SLURM jobs instead of local permutation tests:
# project.prepare_slurm_permutation_test(
#     name="second_feature_set",
#     n_perms=10,
#     conda_env="photonai2.5.2",
#     memory_per_cpu=1,
#     n_jobs=2,
#     run_time="0-00:10:00",
#     random_state=1,
# )

# ---------------------------------------------------------------------
# 4) Compare analyses statistically
# ---------------------------------------------------------------------
# You need to provide n_train and n_test used in your CV setup for Nadeau–Bengio.
# Here we assume a simple split (for demonstration only!).
n_samples = X.shape[0]
n_train = int(0.8 * n_samples)
n_test = n_samples - n_train

# Single pair comparison (Nadeau–Bengio)
project.compare_analyses(
    first_analysis="first_feature_set",
    second_analysis="second_feature_set",
    method="nadeau-bengio",
    n_train=n_train,
    n_test=n_test,
)

# Single pair comparison (permutation-based)
project.compare_analyses(
    first_analysis="all_features",
    second_analysis="second_feature_set",
    method="permutation",
    n_perms=10,
)

# Compare all pairs at once (Nadeau–Bengio)
multi_results_nb = project.compare_multiple_analyses(
    analyses=["all_features", "first_feature_set", "second_feature_set"],
    method="nadeau-bengio",
    n_train=n_train,
    n_test=n_test,
)
print(multi_results_nb)

# Compare all pairs at once (permutation)
multi_results_perm = project.compare_multiple_analyses(
    analyses=["all_features", "first_feature_set", "second_feature_set"],
    method="permutation",
    n_perms=10,
)
print(multi_results_perm)
