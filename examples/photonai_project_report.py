from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold

from photonai.base import Hyperpipe, PipelineElement

from photonai_projects.reporter import PhotonaiProject


def create_pipeline(name: str):
    pipe = Hyperpipe(name,
                     inner_cv=KFold(n_splits=5),
                     outer_cv=KFold(n_splits=5),
                     optimizer='grid_search',
                     metrics=['accuracy', 'precision', 'recall', 'balanced_accuracy'],
                     best_config_metric='balanced_accuracy',
                     project_folder=f'./breast_cancer/{name}')

    pipe += PipelineElement('RandomForestClassifier')
    return pipe


X, y = load_breast_cancer(return_X_y=True)

X_1 = X[:, :3]
X_2 = X[:, 3:6]

for name, current_X in [('all_features', X), ('first_feature_set', X_1), ('second_feature_set', X_2)]:
    pipe = create_pipeline(name)
    pipe.fit(current_X, y)

project = PhotonaiProject(name='breast_cancer', directory='./')
project.collect_results()
project.write_report()
