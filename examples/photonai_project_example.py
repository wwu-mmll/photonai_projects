from photonai_projects.project import PhotonaiProject

from sklearn.datasets import load_breast_cancer


X, y = load_breast_cancer(return_X_y=True)

X_1 = X[:, :3]
X_2 = X[:, 3:6]

project = PhotonaiProject(project_folder='example_project')

for name, current_X in [('all_features', X), ('first_feature_set', X_1), ('second_feature_set', X_2)]:
    project.add_analysis(name=name, X=current_X, y=y,
                         hyperpipe_script='/home/nwinter/PycharmProjects/photonai_projects/examples/hyperpipe_constructor.py',
                         name_hyperpipe_constructor='create_hyperpipe')

project.list_analyses()

#project.run(name='second_feature_set')
#project.run_permutation_test(name='second_feature_set', n_perms=10, overwrite=True)
project.prepare_slurm_permutation_test(name='second_feature_set', n_perms=10,
                                       conda_env='photonai2.5.2', memory_per_cpu=1, n_jobs=2, run_time="0-00:10:00",
                                       random_state=1)