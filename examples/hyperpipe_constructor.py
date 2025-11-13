from photonai import Hyperpipe, PipelineElement
from sklearn.model_selection import KFold


def create_hyperpipe():
    my_pipe = Hyperpipe('',
                        optimizer='grid_search',
                        metrics=['accuracy', 'precision', 'recall'],
                        best_config_metric='accuracy',
                        outer_cv=KFold(n_splits=2),
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
