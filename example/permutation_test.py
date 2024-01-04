import os.path
import uuid
import numpy as np
from sklearn.datasets import load_breast_cancer

from permutation.permutation_test_palma import PermutationTestSetup, PermutationTestPalma


def create_hyperpipe():
    # this is needed here for the parallelisation
    from photonai.base import Hyperpipe, PipelineElement, OutputSettings
    from sklearn.model_selection import GroupKFold
    from sklearn.model_selection import KFold

    settings = OutputSettings(mongodb_connect_url='mongodb://localhost:27017/photon_results')
    my_pipe = Hyperpipe('palma_permutation_test_example',
                        optimizer='grid_search',
                        metrics=['accuracy', 'precision', 'recall'],
                        best_config_metric='accuracy',
                        outer_cv=GroupKFold(n_splits=2),
                        inner_cv=KFold(n_splits=2),
                        calculate_metrics_across_folds=True,
                        use_test_set=True,
                        verbosity=1,
                        project_folder='./tmp/',
                        output_settings=settings)

    # Add transformer elements
    my_pipe += PipelineElement("StandardScaler", hyperparameters={},
                               test_disabled=True, with_mean=True, with_std=True)

    my_pipe += PipelineElement("PCA", test_disabled=False)

    # Add estimator
    my_pipe += PipelineElement("SVC", hyperparameters={'kernel': ['linear', 'rbf']},
                               gamma='scale', max_iter=10000)

    return my_pipe


if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    my_perm_id = "example_permutation_test_perm_id"
    groups = np.random.random_integers(0, 3, (len(y), ))

    # prepare data and scripts for palma
    perm_prep = PermutationTestSetup(X=X, y=y, groups=groups,
                                     hyperpipe_script=os.path.abspath(__file__),
                                     name_hyperpipe_constructor='create_hyperpipe',
                                     perm_id=my_perm_id,
                                     perm_folder='./tmp',
                                     conda_env="lena_resting_state", memory_per_cpu=1, n_jobs=10,
                                     run_time="0-00:10:00", n_perms=1000, random_state=1)
    perm_prep.prepare()

