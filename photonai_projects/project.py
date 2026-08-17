import os
import sys
import shutil
import json
import importlib.util
import typer
from pathlib import Path
from typing import Iterable, Literal, Dict, Optional, Tuple
from scipy import stats
from itertools import combinations

import numpy as np
import pandas as pd

from photonai.processing import ResultsHandler
from photonai.processing.metrics import Scorer

from photonai_projects.utils import find_latest_photonai_run
from photonai_projects.reporter import Reporter


# Records where a sequential permutation test stopped and why.
SEQUENTIAL_STATE_FILE = "sequential_permutation.json"

# Written next to each permutation run once it completes. PHOTONAI's own
# results file holds every fold, config and prediction and runs to several
# megabytes, so re-reading all of them to check a stopping rule is far too
# expensive. This file holds only the mean outer-fold metrics.
RUN_SUMMARY_FILE = "permutation_summary.json"


class PhotonaiProject:
    """
    Manage and compare multiple PHOTONAI analyses within a single project folder.

    This class helps you:
    - create and register new analyses,
    - run PHOTONAI hyperpipes on stored data,
    - run permutation tests (locally or on SLURM), optionally with sequential
      early stopping,
    - aggregate permutation results,
    - compute permutation-based p-values, and
    - statistically compare multiple analyses (Nadeau–Bengio and permutation-based).

    Notes
    -----
    Permutation tests are the expensive part of a project, and most of that cost
    is usually spent confirming that analyses without signal have no signal.
    Passing ``sequential_metric`` to :meth:`run_permutation_test` enables the
    sequential Monte Carlo procedure of Besag and Clifford (1991): sampling
    stops as soon as ``max_exceedances`` permutations have matched or beaten the
    observed value, at which point no number of further permutations could
    produce a small p-value. Analyses that remain significant still use the full
    budget, so power is unaffected — only the null analyses finish early.
    """

    def __init__(
        self,
        project_folder: str,
        feature_importances: bool = False,
    ):
        """
        Initialize a PHOTONAI project.

        Parameters
        ----------
        project_folder : str
            Path to the root folder of the project. All analyses and results are
            stored inside this folder.
        feature_importances : bool, optional
            Whether to compute feature importances (not yet used in this class),
            by default False.
        """
        self.project_folder = project_folder
        self.feature_importances = feature_importances
        self.reporter = Reporter(self.project_folder)
        os.makedirs(self.project_folder, exist_ok=True)

    def run(self, name: str):
        """
        Run a PHOTONAI analysis that has already been added to the project.

        This will:
        - load the hyperpipe constructor from the analysis folder,
        - load the stored data `X.npy` and `y.npy`,
        - fit the hyperpipe, and
        - write PHOTONAI results to the analysis folder.

        Parameters
        ----------
        name : str
            Name of the analysis (subfolder of `project_folder`).

        Returns
        -------
        Hyperpipe
            The fitted PHOTONAI hyperpipe instance.

        Raises
        ------
        ValueError
            If the analysis folder does not exist in the project folder.
        """
        # check that analysis folder exists
        if name not in os.listdir(self.project_folder):
            raise ValueError(
                f"Analysis {name} not found in project folder {self.project_folder}"
            )

        analysis_folder = os.path.join(self.project_folder, name)
        data_folder = os.path.join(analysis_folder, "data")

        pipe = self._load_hyperpipe(analysis_folder, name)
        pipe.output_settings.set_project_folder(analysis_folder)
        pipe.output_settings.set_log_file()
        pipe.name = name
        pipe.project_folder = analysis_folder

        # load data
        X = np.load(os.path.join(data_folder, "X.npy"))
        y = np.load(os.path.join(data_folder, "y.npy"))

        pipe.fit(X, y)

        # if you want to use feature_importances later, you can hook it here
        # if self.feature_importances:
        #     ...

        return pipe

    @staticmethod
    def _load_hyperpipe(analysis_folder: str, name: str, perm_run: bool = False):
        """
        Load and instantiate the hyperpipe constructor for a given analysis.

        The analysis folder must contain:
        - ``hyperpipe_meta.json`` with the key ``"name_hyperpipe_constructor"``.
        - ``hyperpipe_constructor.py`` defining that constructor.

        Parameters
        ----------
        analysis_folder : str
            Path to the analysis folder.
        name : str
            Name of the analysis (used to uniquely name the imported module).
        perm_run : bool, optional
            If True, reduce verbosity of the pipeline (for permutation runs),
            by default False.

        Returns
        -------
        Hyperpipe
            Instantiated PHOTONAI hyperpipe.

        Raises
        ------
        FileNotFoundError
            If required metadata or constructor files are missing.
        KeyError
            If the constructor name is not found in the metadata file.
        AttributeError
            If the constructor function is not found in the constructor module.
        """
        # ------------------------------------------------------------------
        # LOAD HYPERPIPE CONSTRUCTOR FROM HYPERPIPE SCRIPT
        # ------------------------------------------------------------------

        # 1) read metadata to get the constructor function name
        meta_path = os.path.join(analysis_folder, "hyperpipe_meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"No 'hyperpipe_meta.json' found for analysis '{name}' at {meta_path}. "
                f"Did you create this analysis with 'add'?"
            )

        with open(meta_path, "r") as f:
            meta = json.load(f)

        constructor_name = meta.get("name_hyperpipe_constructor", None)
        if constructor_name is None:
            raise KeyError(f"'name_hyperpipe_constructor' not found in {meta_path}")

        # 2) load the hyperpipe_constructor.py as a module
        module_path = os.path.join(analysis_folder, "hyperpipe_constructor.py")
        if not os.path.isfile(module_path):
            raise FileNotFoundError(
                f"No 'hyperpipe_constructor.py' found for analysis '{name}' at {module_path}"
            )

        spec = importlib.util.spec_from_file_location(
            f"hyperpipe_constructor_{name}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, constructor_name):
            raise AttributeError(f"Function '{constructor_name}' not found in {module_path}")

        hyperpipe_constructor = getattr(module, constructor_name)

        # 3) build and run the Hyperpipe
        pipe = hyperpipe_constructor()  # adapt if your constructor needs arguments
        if perm_run:
            pipe.verbosity = -1
        return pipe

    def add(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        hyperpipe_script: str,
        name_hyperpipe_constructor: str,
        **kwargs,
    ):
        """
        Register a new analysis in the project.

        This will:
        - create an analysis subfolder in ``project_folder``,
        - save `X` and `y` as NumPy arrays,
        - copy the hyperpipe script into the analysis folder, and
        - write ``hyperpipe_meta.json`` with the constructor function name.

        Parameters
        ----------
        name : str
            Name of the analysis (subfolder name).
        X : np.ndarray
            Feature matrix with shape (n_samples, n_features).
        y : np.ndarray
            Target vector with shape (n_samples,).
        hyperpipe_script : str
            Path to the Python script that defines the hyperpipe constructor.
        name_hyperpipe_constructor : str
            Name of the hyperpipe constructor function inside `hyperpipe_script`.
        **kwargs :
            Additional keyword arguments (currently unused, reserved for future use).

        Raises
        ------
        ValueError
            If `hyperpipe_script` or `name_hyperpipe_constructor` are not provided.
        """
        if hyperpipe_script is None:
            raise ValueError("hyperpipe_script must be provided in add.")
        if name_hyperpipe_constructor is None:
            raise ValueError("name_hyperpipe_constructor must be provided in add.")

        # create directories for analysis and data
        analysis_folder = os.path.join(self.project_folder, name)
        os.makedirs(analysis_folder, exist_ok=True)
        os.makedirs(os.path.join(analysis_folder, "data"), exist_ok=True)

        # save data to numpy array
        np.save(os.path.join(analysis_folder, "data", "X.npy"), X)
        np.save(os.path.join(analysis_folder, "data", "y.npy"), y)

        # copy script that contains the hyperpipe definition
        shutil.copyfile(
            hyperpipe_script,
            os.path.join(analysis_folder, "hyperpipe_constructor.py"),
        )

        # save metadata (constructor function name etc.)
        meta = {
            "name_hyperpipe_constructor": name_hyperpipe_constructor
            # you could add more fields here (e.g. timestamp, description, etc.)
        }
        meta_path = os.path.join(analysis_folder, "hyperpipe_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def list_analyses(self) -> None:
        """
        Print a list of all analyses available in the project folder.

        The function scans the project folder for subdirectories and prints them
        as available analyses.
        """
        analyses = [
            item
            for item in os.listdir(self.project_folder)
            if os.path.isdir(os.path.join(self.project_folder, item))
        ]
        print("Available PHOTONAI analyses are:")
        for analysis in analyses:
            print(f"  - {analysis}")

    def run_permutation_test(
        self,
        name: str,
        n_perms: int = 1000,
        random_state: int = 15,
        overwrite: bool = False,
        sequential_metric: Optional[str] = None,
        stop_above_p: Optional[float] = None,
        max_exceedances: Optional[int] = None,
    ) -> None:
        """
        Run a local permutation test for a given analysis.

        Parameters
        ----------
        name : str
            Name of the analysis.
        n_perms : int, optional
            Total number of permutation runs, by default 1000.
        random_state : int, optional
            Base random state for generating permutations, by default 15.
        overwrite : bool, optional
            If True, overwrite existing permutation results. If False,
            skip permutations that already have results, by default False.
        sequential_metric : str, optional
            If given, stop early once `max_exceedances` permutation runs have
            reached or beaten the observed value of this metric. An analysis
            without signal reaches that point quickly, so most of the budget is
            spent only on analyses that can still turn out significant. The
            p-value remains valid; see :meth:`sequential_p_value`.
        stop_above_p : float, optional
            Stop as soon as it is clear the p-value is at least this large, and
            never before. ``0.1`` means an analysis that cannot reach p < 0.1
            is abandoned, while anything still able to is run to the full
            budget. Give this or `max_exceedances`, not both.
        max_exceedances : int, optional
            The same rule expressed as an exceedance count; equals
            ``stop_above_p * n_perms``.
        """
        perm_runs = range(n_perms)
        self._run_permutation_test(
            name=name,
            random_state=random_state,
            n_perms=n_perms,
            overwrite=overwrite,
            perm_runs=perm_runs,
            sequential_metric=sequential_metric,
            stop_above_p=stop_above_p,
            max_exceedances=max_exceedances,
        )

    def check_permutation_test(
        self,
        name: str,
        n_perms: int = 1000,
    ):
        """
        Check which permutation runs have a stored PHOTONAI results file.

        Parameters
        ----------
        name : str
            Name of the analysis.
        n_perms : int, optional
            Expected number of permutation runs, by default 1000.

        Returns
        -------
        list of int
            Sorted list of permutation run indices that were found.
        list of int
            Sorted list of permutation run indices that are missing.
        """
        perm_runs = range(n_perms)
        perm_folder = Path(self.project_folder) / name / "permutations"

        found_runs = [
            int(folder.name)
            for folder in perm_folder.iterdir()
            if folder.is_dir() and (folder / "photonai_results.json").exists()
        ]
        missing_runs = sorted(set(perm_runs) - set(found_runs))
        print(
            f"Found {len(found_runs)} permutation runs, {len(missing_runs)} are missing."
        )
        return sorted(found_runs), missing_runs

    def _load_true_fold_results(self, name: str) -> pd.DataFrame:
        """
        Load per-outer-fold performance metrics for an analysis.

        Parameters
        ----------
        name : str
            Name of the analysis.

        Returns
        -------
        pandas.DataFrame
            DataFrame where rows correspond to outer folds and columns to metrics.

        Raises
        ------
        FileNotFoundError
            If no PHOTONAI run can be found for the given analysis.
        """
        photonai_folder = find_latest_photonai_run(Path(self.project_folder) / name)
        if photonai_folder is None:
            raise FileNotFoundError(
                f"No PHOTONAI run found for analysis {name} in {self.project_folder}"
            )

        handler = ResultsHandler()
        handler.load_from_file(str(Path(photonai_folder) / "photonai_results.json"))
        return pd.DataFrame(handler.get_performance_outer_folds())

    def _load_true_results(self, name: str) -> pd.Series:
        """
        Load mean performance metrics across outer folds for an analysis.

        Parameters
        ----------
        name : str
            Name of the analysis.

        Returns
        -------
        pandas.Series
            Series of mean metric values indexed by metric name.
        """
        folds_df = self._load_true_fold_results(name)
        return folds_df.mean(axis=0)

    def _ensure_and_load_permutation_results(
        self,
        name: str,
        n_perms: int = 1000,
    ) -> pd.DataFrame:
        """
        Ensure that aggregated permutation results exist and load them.

        If ``permutation_results.csv`` is missing, it is created by calling
        :meth:`aggregate_permutation_test`.

        Parameters
        ----------
        name : str
            Name of the analysis.
        n_perms : int, optional
            Number of permutations expected, by default 1000.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing aggregated permutation results with a ``run`` column.
        """
        perm_results_file = Path(self.project_folder) / name / "permutation_results.csv"
        if not perm_results_file.exists():
            self.aggregate_permutation_test(name, n_perms)
        return pd.read_csv(perm_results_file)

    # -------------------------------------------------
    # Sequential permutation testing
    # -------------------------------------------------
    @staticmethod
    def resolve_exceedance_budget(n_perms: int,
                                  stop_above_p: Optional[float] = None,
                                  max_exceedances: Optional[int] = None) -> int:
        """
        Translate a stopping threshold into an exceedance budget.

        Sampling can only stop early once the p-value has reached
        ``max_exceedances / n_perms``, so the two parameters are the same thing
        expressed differently::

            max_exceedances = stop_above_p * n_perms

        Choosing ``stop_above_p=0.1`` with 1000 permutations therefore means
        "stop as soon as it is clear the p-value is at least 0.1, and never
        before", which is usually the way one wants to think about it.

        Parameters
        ----------
        n_perms : int
            Total number of permutations planned.
        stop_above_p : float, optional
            The p-value above which sampling may stop. Must lie in (0, 1].
        max_exceedances : int, optional
            The exceedance budget, given directly.

        Returns
        -------
        int
            The exceedance budget to use, at least 1.

        Raises
        ------
        ValueError
            If both or neither parameter is given, or `stop_above_p` is outside
            (0, 1].
        """
        if (stop_above_p is None) == (max_exceedances is None):
            raise ValueError("Give exactly one of 'stop_above_p' or "
                             "'max_exceedances'.")

        if max_exceedances is not None:
            if max_exceedances < 1:
                raise ValueError("max_exceedances must be at least 1.")
            return int(max_exceedances)

        if not 0 < stop_above_p <= 1:
            raise ValueError(f"stop_above_p must be in (0, 1], got {stop_above_p}.")

        # ceil, so the realised threshold is never below the one requested
        return max(1, int(np.ceil(stop_above_p * n_perms)))

    @staticmethod
    def _is_at_least_as_extreme(observed: float,
                                permuted: np.ndarray,
                                greater_is_better: bool) -> np.ndarray:
        """
        Flag permutation results that are at least as extreme as the observed one.

        Missing values count as extreme. A permutation run whose metric could
        not be computed is treated as evidence against the alternative, which
        keeps the resulting p-value conservative.

        Parameters
        ----------
        observed : float
            Metric value obtained with the true targets.
        permuted : numpy.ndarray
            Metric values obtained under permutation.
        greater_is_better : bool
            Whether larger values of the metric indicate better performance.

        Returns
        -------
        numpy.ndarray of bool
            One flag per permutation run.
        """
        permuted = np.asarray(permuted, dtype=float)
        missing = np.isnan(permuted)

        if greater_is_better:
            extreme = permuted >= observed
        else:
            extreme = permuted <= observed

        return extreme | missing

    @staticmethod
    def sequential_p_value(observed: float,
                           permuted: Iterable[float],
                           greater_is_better: bool,
                           max_exceedances: int = 20,
                           n_perms: int = 1000) -> Dict:
        """
        Sequential Monte Carlo p-value after Besag and Clifford (1991).

        Permutation results are examined in the order they were generated and
        counted whenever they are at least as extreme as the observed value.
        Sampling stops as soon as `max_exceedances` such results have appeared,
        because at that point the analysis cannot reach a small p-value however
        many further permutations are drawn.

        If sampling stopped early at the ``L``-th permutation, the p-value is
        ``max_exceedances / L``. Otherwise it is the usual
        ``(1 + exceedances) / (1 + n_perms)``. Both are valid p-values under the
        null hypothesis, so the saving in computation costs no validity.

        Permutations that were planned but never ran are counted as exceedances,
        matching the conservative treatment of failed runs elsewhere in this
        class.

        Parameters
        ----------
        observed : float
            Metric value obtained with the true targets.
        permuted : iterable of float
            Metric values under permutation, **in the order they were run**.
            The order matters: it determines where sampling would have stopped.
        greater_is_better : bool
            Whether larger values of the metric indicate better performance.
        max_exceedances : int, optional
            Number of exceedances at which sampling stops, by default 20.
            Larger values give a more precise p-value near the stopping region
            at the cost of more permutations.
        n_perms : int, optional
            Total number of permutations planned, by default 1000.

        Returns
        -------
        dict
            With keys ``p_value``, ``stopped_early``, ``n_exceedances``,
            ``n_perms_used`` (how many permutations were needed) and
            ``n_perms_planned``.

        References
        ----------
        Besag, J. and Clifford, P. (1991). Sequential Monte Carlo p-values.
        Biometrika, 78(2), 301-304.

        Examples
        --------
        A clearly null analysis stops long before the full budget:

        >>> import numpy as np
        >>> permuted = np.linspace(-0.05, 0.05, 1000)
        >>> result = PhotonaiProject.sequential_p_value(
        ...     observed=0.0, permuted=permuted, greater_is_better=True,
        ...     max_exceedances=20, n_perms=1000)
        >>> result['stopped_early']
        True
        """
        if max_exceedances < 1:
            raise ValueError("max_exceedances must be at least 1.")

        permuted = np.asarray(list(permuted), dtype=float)

        extreme = PhotonaiProject._is_at_least_as_extreme(
            observed, permuted, greater_is_better)
        cumulative = np.cumsum(extreme)

        reached = np.flatnonzero(cumulative >= max_exceedances)
        if reached.size:
            # +1 converts the zero-based position into a count of permutations
            n_used = int(reached[0]) + 1
            return {'p_value': max_exceedances / n_used,
                    'stopped_early': True,
                    'n_exceedances': int(max_exceedances),
                    'n_perms_used': n_used,
                    'n_perms_planned': int(n_perms)}

        # budget never exhausted: fall back to the standard estimator, counting
        # permutations that were planned but never ran as exceedances
        observed_exceedances = int(cumulative[-1]) if cumulative.size else 0
        never_ran = max(0, n_perms - permuted.size)
        total = observed_exceedances + never_ran

        return {'p_value': (1 + total) / (1 + n_perms),
                'stopped_early': False,
                'n_exceedances': observed_exceedances,
                'n_perms_used': int(permuted.size),
                'n_perms_planned': int(n_perms)}

    def sequential_status(self,
                          name: str,
                          metric: str,
                          max_exceedances: int = 20,
                          n_perms: int = 1000) -> Dict:
        """
        Report whether an analysis has already accumulated enough exceedances.

        Reads the permutation runs computed so far and decides whether further
        permutations can still change the conclusion. Use it between batches of
        a staged permutation test to decide whether to submit the next batch.

        Parameters
        ----------
        name : str
            Name of the analysis.
        metric : str
            Metric the stopping rule is applied to, e.g. ``explained_variance``.
        max_exceedances : int, optional
            Number of exceedances at which sampling stops, by default 20.
        n_perms : int, optional
            Total number of permutations planned, by default 1000.

        Returns
        -------
        dict
            The result of :meth:`sequential_p_value` for the runs completed so
            far, plus ``metric`` and ``should_continue``.
        """
        true_results = self._load_true_results(name)
        if metric not in true_results.index:
            raise KeyError(f"Metric '{metric}' not among the analysis metrics: "
                           f"{list(true_results.index)}")

        runs = self._collect_permutation_runs(name)
        permuted = (runs.sort_values('run')[metric].to_numpy()
                    if not runs.empty else np.array([]))

        status = self.sequential_p_value(
            observed=float(true_results[metric]),
            permuted=permuted,
            greater_is_better=Scorer.greater_is_better_distinction(metric),
            max_exceedances=max_exceedances,
            n_perms=n_perms)

        status['metric'] = metric
        status['should_continue'] = (not status['stopped_early']
                                     and status['n_perms_used'] < n_perms)
        return status

    def _write_sequential_state(self, name: str, status: Dict) -> None:
        """
        Persist the sequential stopping decision for an analysis.

        Parameters
        ----------
        name : str
            Name of the analysis.
        status : dict
            Result of :meth:`sequential_status`.
        """
        path = Path(self.project_folder) / name / SEQUENTIAL_STATE_FILE
        with open(path, 'w') as file:
            json.dump(status, file, indent=2)

    def read_sequential_state(self, name: str) -> Optional[Dict]:
        """
        Read the stored sequential stopping decision, if there is one.

        Parameters
        ----------
        name : str
            Name of the analysis.

        Returns
        -------
        dict or None
            The stored state, or None if the analysis was not run sequentially.
        """
        path = Path(self.project_folder) / name / SEQUENTIAL_STATE_FILE
        if not path.exists():
            return None
        with open(path, 'r') as file:
            return json.load(file)

    # -------------------------------------------------
    # Permutation aggregation / p-values
    # -------------------------------------------------
    @staticmethod
    def _summarize_run(run_folder: Path) -> Optional[pd.Series]:
        """
        Read one permutation run's mean outer-fold metrics.

        Prefers the small summary file. If it is absent — because the run
        predates summaries, or was written by an older version — the full
        PHOTONAI results file is parsed once and the summary is written for
        next time.

        Parameters
        ----------
        run_folder : pathlib.Path
            Folder of a single permutation run.

        Returns
        -------
        pandas.Series or None
            Mean metrics with a ``run`` entry, or None if the run has no
            results yet.
        """
        summary_file = run_folder / RUN_SUMMARY_FILE
        if summary_file.exists():
            try:
                with open(summary_file, "r") as file:
                    return pd.Series(json.load(file))
            except (json.JSONDecodeError, OSError):
                # a truncated summary (e.g. a job killed mid-write) is not
                # worth failing over; fall through and rebuild it
                pass

        results_file = run_folder / "photonai_results.json"
        if not results_file.exists():
            return None

        handler = ResultsHandler()
        handler.load_from_file(str(results_file))
        metrics = pd.DataFrame(handler.get_performance_outer_folds()).mean(axis=0)
        metrics["run"] = int(run_folder.name)

        PhotonaiProject._write_run_summary(run_folder, metrics)
        return metrics

    @staticmethod
    def _write_run_summary(run_folder: Path, metrics: pd.Series) -> None:
        """
        Write the small per-run summary used by the sequential stopping rule.

        Parameters
        ----------
        run_folder : pathlib.Path
            Folder of a single permutation run.
        metrics : pandas.Series
            Mean outer-fold metrics, including a ``run`` entry.
        """
        try:
            with open(run_folder / RUN_SUMMARY_FILE, "w") as file:
                json.dump({key: float(value) for key, value in metrics.items()},
                          file, indent=2)
        except OSError:
            # the summary is a cache, not a result: never fail a run over it
            pass

    def _collect_permutation_runs(self, name: str) -> pd.DataFrame:
        """
        Load the mean outer-fold metrics of every completed permutation run.

        Reads the per-run summary files rather than PHOTONAI's full results,
        which makes the sequential stopping check cheap enough to run after
        every permutation.

        Parameters
        ----------
        name : str
            Name of the analysis.

        Returns
        -------
        pandas.DataFrame
            One row per completed run with a ``run`` column, sorted by run
            index. Empty if no run has completed.
        """
        perm_folder = Path(self.project_folder) / name / "permutations"
        if not perm_folder.exists():
            return pd.DataFrame()

        rows = []
        for folder in sorted(perm_folder.iterdir(), key=lambda f: f.name):
            if not folder.is_dir():
                continue
            metrics = self._summarize_run(folder)
            if metrics is not None:
                rows.append(metrics)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("run").reset_index(drop=True)

    def aggregate_permutation_test(self, name: str, n_perms: int = 1000) -> None:
        """
        Aggregate results from individual permutation runs into a single CSV file.

        This function:
        - collects mean outer-fold metrics for each permutation run,
        - ensures that all permutation indices `0..n_perms-1` are represented,
        - fills missing values with ±∞ depending on whether higher is better, and
        - writes the result to ``permutation_results.csv`` in the analysis folder.

        Parameters
        ----------
        name : str
            Name of the analysis.
        n_perms : int, optional
            Number of permutation runs, by default 1000.
        """
        perm_results = self._collect_permutation_runs(name)

        # Ensure all runs 0..n_perms-1 are represented
        df_perm_index = pd.DataFrame(
            np.arange(n_perms), columns=["run"], index=np.arange(n_perms)
        )
        perm_results = pd.merge(df_perm_index, perm_results, on="run", how="left")

        for metric in list(perm_results.keys()):
            if metric == "run":
                continue
            greater_is_better = Scorer.greater_is_better_distinction(metric)
            if greater_is_better:
                perm_results[metric] = perm_results[metric].fillna(np.inf)
            else:
                perm_results[metric] = perm_results[metric].fillna(-np.inf)

        perm_results.to_csv(
            Path(self.project_folder) / name / "permutation_results.csv", index=False
        )

    def calculate_permutation_p_values(
        self,
        name: str,
        n_perms: int = 1000,
    ) -> None:
        """
        Compute permutation-based p-values for a given analysis.

        For each metric, this function compares the true mean performance to the
        distribution of permutation results and computes a one-sided p-value
        using the standard (k+1)/(n_perms+1) formulation.

        Parameters
        ----------
        name : str
            Name of the analysis.
        n_perms : int, optional
            Number of permutation runs, by default 1000.
        """
        true_results = self._load_true_results(name)

        sequential_state = self.read_sequential_state(name)
        if sequential_state is not None and sequential_state["stopped_early"]:
            # The run was cut short on purpose, so the missing permutations must
            # not be counted as failures. Only the metric the stopping rule was
            # applied to has a meaningful p-value here: sampling stopped when
            # *that* metric ran out of budget, which says nothing about the
            # others.
            metric_name = sequential_state["metric"]
            print(
                f"'{name}' was stopped sequentially after "
                f"{sequential_state['n_perms_used']} of "
                f"{sequential_state['n_perms_planned']} permutations. "
                f"Reporting the sequential p-value for '{metric_name}' only."
            )
            pd.DataFrame({metric_name: sequential_state["p_value"]},
                         index=[0]).to_csv(
                Path(self.project_folder) / name / "permutation_p_values.csv",
                index=False,
            )
            return

        perm_results = self._ensure_and_load_permutation_results(name, n_perms)

        p_values: Dict[str, float] = {}
        for metric in list(true_results.keys()):
            greater_is_better = Scorer.greater_is_better_distinction(metric)
            current_perm_results = np.asarray(perm_results[metric], dtype=float)

            if greater_is_better:
                current_perm_results[np.isnan(current_perm_results)] = np.inf
                p_values[metric] = (
                    np.sum(true_results[metric] < current_perm_results) + 1
                ) / (n_perms + 1)
            else:
                current_perm_results[np.isnan(current_perm_results)] = -np.inf
                p_values[metric] = (
                    np.sum(true_results[metric] > current_perm_results) + 1
                ) / (n_perms + 1)

            n_valid = n_perms - np.sum(np.isinf(current_perm_results))
            print(
                f"p-value for {metric}: {p_values[metric]} "
                f"(based on n={n_valid} valid permutations)"
            )

        pd.DataFrame(p_values, index=[0]).to_csv(
            Path(self.project_folder) / name / "permutation_p_values.csv",
            index=False,
        )

    # -------------------------------------------------
    # Nadeau–Bengio helper
    # -------------------------------------------------
    @staticmethod
    def _nadeau_bengio_p_value(
        diffs: np.ndarray,
        n_train: int,
        n_test: int,
    ) -> Tuple[float, float]:
        """
        Two-sided Nadeau & Bengio corrected resampled t-test.

        Parameters
        ----------
        diffs : np.ndarray
            Array of per-fold score differences (analysis2 - analysis1).
        n_train : int
            Number of training samples used in each resample.
        n_test : int
            Number of test samples used in each resample.

        Returns
        -------
        float
            Two-sided p-value of the test.
        float
            t-statistic of the corrected t-test.

        Notes
        -----
        The corrected variance is computed as:

        .. math::

            \\text{Var}_c = \\left(\\frac{1}{k} + \\frac{n_{test}}{n_{train}}\\right) s^2
        """
        diffs = np.asarray(diffs, dtype=float)
        k = len(diffs)
        if k < 2:
            return 1.0, 0.0  # not enough folds

        mean_diff = np.mean(diffs)
        var_diff = np.var(diffs, ddof=1)
        rho = n_test / n_train
        corrected_var = (1.0 / k + rho) * var_diff
        if corrected_var <= 0:
            return 1.0, 0.0

        t_stat = mean_diff / np.sqrt(corrected_var)
        df = k - 1

        # two-sided p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        return p_value, t_stat

    # -------------------------------------------------
    # Comparison of two analyses
    # -------------------------------------------------
    def compare_analyses(
        self,
        first_analysis: str,
        second_analysis: str,
        method: Literal["nadeau-bengio", "permutation"] = "nadeau-bengio",
        metric: str | None = None,
        n_perms: int = 1000,
        n_train: int | None = None,
        n_test: int | None = None,
        print_report: bool = True,
    ) -> pd.DataFrame:
        """
        Compare two analyses using statistical tests.

        You can choose between:
        - Nadeau–Bengio corrected t-test on outer-fold scores, or
        - permutation-based null distribution of performance differences.

        Parameters
        ----------
        first_analysis : str
            Name of the first analysis.
        second_analysis : str
            Name of the second analysis.
        method : {"nadeau-bengio", "permutation"}, optional
            Statistical comparison method, by default "nadeau-bengio".
        metric : str or None, optional
            If given, only compare this metric. If None, compare all metrics
            common to both analyses, by default None.
        n_perms : int, optional
            Number of permutation runs (only for permutation-based comparison),
            by default 1000.
        n_train : int or None, optional
            Number of training samples used during cross-validation (required
            for Nadeau–Bengio), by default None.
        n_test : int or None, optional
            Number of test samples used during cross-validation (required
            for Nadeau–Bengio), by default None.
        print_report : bool, optional
            If True, print a formatted comparison report, by default True.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by metric, containing columns such as:
            ``p_value``, ``effect``, and method-specific fields (e.g. ``t_stat``,
            ``n_folds`` or ``n_valid_perms``).

        Raises
        ------
        ValueError
            If an invalid method is passed or required parameters are missing.
        """
        valid_methods = {"nadeau-bengio", "permutation"}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Valid options are: {valid_methods}"
            )

        results: list[dict] = []

        # ---------------- permutation-based comparison ----------------
        if method == "permutation":
            # Load true and permutation results for both analyses
            true1 = self._load_true_results(first_analysis)
            perm1 = self._ensure_and_load_permutation_results(
                first_analysis, n_perms
            )

            true2 = self._load_true_results(second_analysis)
            perm2 = self._ensure_and_load_permutation_results(
                second_analysis, n_perms
            )

            # sanity check: runs aligned
            if not np.array_equal(perm1["run"].values, perm2["run"].values):
                raise ValueError(
                    "Permutation indices (run column) do not match between analyses."
                )

            if metric is None:
                metrics = set(true1.index).intersection(true2.index)
            else:
                metrics = [metric]
            for metric in metrics:
                greater_is_better = Scorer.greater_is_better_distinction(metric)

                # true difference: analysis2 - analysis1
                true_diff = float(true2[metric] - true1[metric])

                # permutation differences per run
                perm_diff = (
                    np.asarray(perm2[metric], dtype=float)
                    - np.asarray(perm1[metric], dtype=float)
                )

                if greater_is_better:
                    perm_diff[np.isnan(perm_diff)] = np.inf
                    p_val = (np.sum(true_diff < perm_diff) + 1) / (n_perms + 1)
                else:
                    perm_diff[np.isnan(perm_diff)] = -np.inf
                    p_val = (np.sum(true_diff > perm_diff) + 1) / (n_perms + 1)

                n_valid = n_perms - np.sum(np.isinf(perm_diff))
                print(
                    f"[permutation] {metric}: p={p_val}, "
                    f"true_diff={true_diff} (n_valid={n_valid})"
                )

                results.append(
                    {
                        "metric": metric,
                        "method": "permutation",
                        "p_value": p_val,
                        "effect": true_diff,  # analysis2 - analysis1
                        "n_valid_perms": int(n_valid),
                    }
                )

        # ---------------- Nadeau–Bengio comparison ----------------
        elif method == "nadeau-bengio":
            if n_train is None or n_test is None:
                raise ValueError(
                    "n_train and n_test must be provided for the Nadeau-Bengio test."
                )

            folds1 = self._load_true_fold_results(first_analysis)
            folds2 = self._load_true_fold_results(second_analysis)

            if metric is None:
                metrics = set(folds1.columns).intersection(folds2.columns)
            else:
                metrics = [metric]
            for metric in metrics:
                # fold-wise differences: analysis2 - analysis1
                diffs = folds2[metric].values - folds1[metric].values
                p_val, t_stat = self._nadeau_bengio_p_value(
                    diffs,
                    n_train=n_train,
                    n_test=n_test,
                )
                mean_diff = float(np.mean(diffs))

                print(
                    f"[nadeau-bengio] {metric}: p={p_val}, t={t_stat}, "
                    f"A={folds1[metric].mean()}[{folds1[metric].std()}], "
                    f"B={folds2[metric].mean()}[{folds2[metric].std()}], "
                    f"mean_diff={mean_diff}"
                )

                results.append(
                    {
                        "metric": metric,
                        "method": "nadeau-bengio",
                        "p_value": p_val,
                        "t_stat": t_stat,
                        "effect": mean_diff,  # analysis2 - analysis1
                        "n_folds": len(diffs),
                    }
                )

        df = pd.DataFrame(results).set_index("metric")
        if print_report:
            self.print_comparison_report(first_analysis, second_analysis, df)
        return df

    def print_comparison_report(
        self,
        first_analysis: str,
        second_analysis: str,
        results_df: pd.DataFrame,
    ) -> None:
        """
        Print a formatted summary for the comparison of two analyses.

        This report includes, for each metric:
        - mean and standard deviation of the true performance for both analyses,
        - the difference (second - first),
        - the statistical method, and
        - method-specific statistics (p-value, t-statistic, etc.).

        Parameters
        ----------
        first_analysis : str
            Name of the first analysis.
        second_analysis : str
            Name of the second analysis.
        results_df : pandas.DataFrame
            Output DataFrame from :meth:`compare_analyses`.
        """
        # Load true per-fold results to get mean & std
        folds1 = self._load_true_fold_results(first_analysis)
        folds2 = self._load_true_fold_results(second_analysis)

        print("\n" + "=" * 80)
        print(f"COMPARISON REPORT: {first_analysis}  vs  {second_analysis}")
        print("=" * 80)

        for _, row in results_df.reset_index().iterrows():
            metric = row["metric"]
            method = row["method"]

            true1 = folds1[metric]
            true2 = folds2[metric]

            mean1, std1 = true1.mean(), true1.std(ddof=1)
            mean2, std2 = true2.mean(), true2.std(ddof=1)

            diff = mean2 - mean1

            print(f"\n--- Metric: {metric} ---")
            print(f"{first_analysis}: mean={mean1:.4f}, std={std1:.4f}")
            print(f"{second_analysis}: mean={mean2:.4f}, std={std2:.4f}")
            print(f"Difference (second - first): {diff:.4f}")

            print(f"\nMethod: {method}")

            if method == "nadeau-bengio":
                print(f"T-statistic: {row.get('t_stat', float('nan')):.4f}")
                print(f"P-value:     {row['p_value']:.6f}")

            elif method == "permutation":
                print(f"P-value:     {row['p_value']:.6f}")
                print(f"Valid perms: {row.get('n_valid_perms', 'N/A')}")

            print("-" * 80)

        print("\n")

    def compare_multiple_analyses(
        self,
        analyses: Iterable[str],
        method: Literal["nadeau-bengio", "permutation"] = "nadeau-bengio",
        metric: str | None = None,
        n_perms: int = 1000,
        n_train: int | None = None,
        n_test: int | None = None,
    ) -> pd.DataFrame:
        """
        Compare all pairs of analyses using :meth:`compare_analyses`.

        Parameters
        ----------
        analyses : iterable of str
            Names of analyses (e.g. ``["A", "B", "C", "D"]``).
        method : {"nadeau-bengio", "permutation"}, optional
            Which comparison method to use, by default "nadeau-bengio".
        metric : str or None, optional
            If given, only compare this metric. If None, compare all metrics
            common to each pair, by default None.
        n_perms : int, optional
            Number of permutations (for permutation-based comparison),
            by default 1000.
        n_train : int, optional
            Number of training samples (for Nadeau–Bengio).
        n_test : int, optional
            Number of test samples (for Nadeau–Bengio).

        Returns
        -------
        pandas.DataFrame
            Long-format table with one row per (metric, pair), including
            p-values, effect sizes, and method-specific statistics.

        Raises
        ------
        ValueError
            If fewer than two analyses are provided.
        """
        analyses = list(analyses)
        if len(analyses) < 2:
            raise ValueError("Need at least two analyses to compare.")

        all_results = []

        for first, second in combinations(analyses, 2):
            print(f"Comparing '{first}' vs '{second}' using {method}...")
            pair_df = self.compare_analyses(
                first_analysis=first,
                second_analysis=second,
                method=method,
                metric=metric,
                n_perms=n_perms,
                n_train=n_train,
                n_test=n_test,
                print_report=False,
            )

            # Make sure we don't accidentally mutate the original
            pair_df = pair_df.copy()
            pair_df["first_analysis"] = first
            pair_df["second_analysis"] = second

            # move metric from index to column for stacking
            pair_df = pair_df.reset_index()  # 'metric' becomes a column
            all_results.append(pair_df)

        if not all_results:
            return pd.DataFrame()

        result_df = pd.concat(all_results, ignore_index=True)

        return result_df

    def _run_permutation_test(
        self,
        name: str,
        random_state: int = 15,
        n_perms: int = 1000,
        overwrite: bool = False,
        perm_runs: range = range(1000),
        sequential_metric: Optional[str] = None,
        stop_above_p: Optional[float] = None,
        max_exceedances: Optional[int] = None,
    ) -> None:
        """
        Internal helper to run a subset of permutation tests for an analysis.

        Parameters
        ----------
        name : str
            Name of the analysis.
        random_state : int, optional
            Base random state for permutation generation, by default 15.
        n_perms : int, optional
            Total number of permutation runs, by default 1000.
        overwrite : bool, optional
            Whether to overwrite existing permutation results, by default False.
        perm_runs : range, optional
            Iterable of permutation indices to run, by default range(1000).
        sequential_metric : str, optional
            Metric the sequential stopping rule is applied to. If None, all
            requested permutations are computed.
        stop_above_p : float, optional
            p-value above which sampling may stop.
        max_exceedances : int, optional
            The same rule expressed as an exceedance count.

        Raises
        ------
        ValueError
            If the analysis folder does not exist.
        """
        # check that analysis folder exists
        if name not in os.listdir(self.project_folder):
            raise ValueError(
                f"Analysis {name} not found in project folder {self.project_folder}"
            )

        analysis_folder = os.path.join(self.project_folder, name)
        data_folder = os.path.join(analysis_folder, "data")
        perm_folder = os.path.join(analysis_folder, "permutations")

        if sequential_metric is not None:
            max_exceedances = self.resolve_exceedance_budget(
                n_perms, stop_above_p, max_exceedances)

        # load data
        X = np.load(os.path.join(data_folder, "X.npy"))
        y = np.load(os.path.join(data_folder, "y.npy"))

        # a previous batch may already have settled the question
        if sequential_metric is not None:
            status = self.sequential_status(name, sequential_metric,
                                            max_exceedances, n_perms)
            if status["stopped_early"]:
                print(
                    f"Sequential stopping: '{name}' already reached "
                    f"{max_exceedances} exceedances of {sequential_metric} after "
                    f"{status['n_perms_used']} permutations "
                    f"(p = {status['p_value']:.4f}). Skipping this batch."
                )
                self._write_sequential_state(name, status)
                return

        for perm_run in perm_runs:
            current_perm_folder = os.path.join(perm_folder, str(perm_run))
            if (
                not overwrite
                and os.path.exists(
                    os.path.join(current_perm_folder, "photonai_results.json")
                )
            ):
                print(
                    f"Skipping permutation {perm_run + 1}/{n_perms} as it already exists."
                )
                continue

            print(f"Running permutation {perm_run + 1}/{n_perms}")
            np.random.seed(random_state + perm_run)
            y_perm = np.random.permutation(y)
            pipe = self._load_hyperpipe(analysis_folder, name, perm_run=True)
            pipe.output_settings.set_project_folder(
                os.path.join(perm_folder, str(perm_run))
            )
            pipe.output_settings.set_log_file()
            pipe.name = name
            pipe.project_folder = os.path.join(perm_folder, str(perm_run))
            pipe.fit(X, y_perm)
            shutil.copyfile(
                os.path.join(
                    pipe.output_settings.results_folder, "photonai_results.json"
                ),
                os.path.join(
                    os.path.join(perm_folder, str(perm_run)),
                    "photonai_results.json",
                ),
            )
            shutil.rmtree(pipe.output_settings.results_folder)

            # write the small summary now, while the results are already loaded
            self._summarize_run(Path(current_perm_folder))

            if sequential_metric is not None:
                status = self.sequential_status(name, sequential_metric,
                                                max_exceedances, n_perms)
                self._write_sequential_state(name, status)
                if status["stopped_early"]:
                    print(
                        f"Sequential stopping: {max_exceedances} exceedances of "
                        f"{sequential_metric} reached after "
                        f"{status['n_perms_used']} of {n_perms} permutations "
                        f"(p = {status['p_value']:.4f}). Stopping."
                    )
                    return

    def run_permutation_test_slurm(
        self,
        name: str,
        n_perms: int = 1000,
        random_state: int = 15,
        overwrite: bool = False,
        slurm_job_id: int | None = None,
        n_perms_per_job: int | None = None,
        stage: int = 1,
        n_jobs_per_stage: int = 0,
        sequential_metric: Optional[str] = None,
        stop_above_p: Optional[float] = None,
        max_exceedances: Optional[int] = None,
    ) -> None:
        """
        Run a subset of permutation tests for use in a SLURM array job.

        Parameters
        ----------
        name : str
            Name of the analysis.
        n_perms : int, optional
            Total number of permutation runs, by default 1000.
        random_state : int, optional
            Base random state for permutation generation, by default 15.
        overwrite : bool, optional
            Whether to overwrite existing permutation results, by default False.
        slurm_job_id : int or None, optional
            Index of the SLURM array job (starting at 1).
        n_perms_per_job : int or None, optional
            Number of permutations to run in this job.
        stage : int, optional
            1-based stage index for staged runs, by default 1.
        n_jobs_per_stage : int, optional
            Array size of one stage; needed to make permutation indices unique
            across stages. Defaults to 0, meaning a single unstaged array.
        sequential_metric : str, optional
            Metric the sequential stopping rule is applied to. Array tasks that
            start after the budget has been exhausted exit immediately, so the
            saving grows with how much of the array is still queued.
        stop_above_p : float, optional
            p-value above which sampling may stop.
        max_exceedances : int, optional
            The same rule expressed as an exceedance count.
        """
        # In a staged run the array restarts at 1 each stage, so the stage
        # offset is what makes the permutation indices globally unique.
        global_job_id = (stage - 1) * n_jobs_per_stage + slurm_job_id if n_jobs_per_stage else slurm_job_id
        perms_to_do = np.arange(
            (global_job_id - 1) * n_perms_per_job,
            (global_job_id - 1) * n_perms_per_job + n_perms_per_job,
        )
        perms_to_do = perms_to_do[perms_to_do < n_perms]
        self._run_permutation_test(
            name=name,
            random_state=random_state,
            n_perms=n_perms,
            overwrite=overwrite,
            perm_runs=perms_to_do,
            sequential_metric=sequential_metric,
            stop_above_p=stop_above_p,
            max_exceedances=max_exceedances,
        )

    def prepare_slurm_permutation_test(
        self,
        name: str,
        n_perms: int,
        conda_env: str,
        memory_per_cpu: int,
        n_jobs: int,
        run_time: str = "0-01:00:00",
        random_state: int = 1,
        sequential_metric: Optional[str] = None,
        stop_above_p: Optional[float] = None,
        max_exceedances: Optional[int] = None,
    ) -> None:
        """
        Prepare a SLURM job script for running permutation tests in parallel.

        This function:
        - computes how many permutations each SLURM array job should run,
        - copies the current project script into the project folder, and
        - writes a SLURM script that calls :func:`run_perm_job`.

        Parameters
        ----------
        name : str
            Name of the analysis.
        n_perms : int
            Total number of permutation runs.
        conda_env : str
            Name of the conda environment to activate in the SLURM job.
        memory_per_cpu : int
            Memory per CPU in GB.
        n_jobs : int
            Number of jobs in the SLURM array.
        run_time : str, optional
            Maximum wall time for each job (SLURM time format),
            by default "0-01:00:00".
        random_state : int, optional
            Base random state, by default 1.
        sequential_metric : str, optional
            If given, the generated script enables sequential stopping on this
            metric. Submit the array in stages for the largest saving: array
            tasks check the exceedance budget before doing any work, so any task
            still queued when the budget is spent exits immediately.
        max_exceedances : int, optional
            Exceedance budget for sequential stopping, by default 20.

        Raises
        ------
        ValueError
            If the analysis folder does not exist in the project folder.
        """
        if name not in os.listdir(self.project_folder):
            raise ValueError(
                f"Analysis {name} not found in project folder {self.project_folder}"
            )

        analysis_folder = os.path.join(self.project_folder, name)
        # calculate the number of perms per job
        n_perms_per_job = int(n_perms / n_jobs)

        sequential_arguments = ""
        if sequential_metric is not None:
            budget = self.resolve_exceedance_budget(n_perms, stop_above_p,
                                                    max_exceedances)
            sequential_arguments = (
                f" --sequential-metric {sequential_metric}"
                f" --max-exceedances {budget}"
            )

        # copy script that contains the permutation test
        shutil.copyfile(
            os.path.abspath(__file__),
            os.path.join(self.project_folder, os.path.basename(__file__)),
        )

        # create slurm script
        cmd = f"""#!/bin/bash

#SBATCH --job-name={name + "_perm_test"}
#SBATCH --output=logs/job_%a.log

#SBATCH --partition normal
#SBATCH --mem-per-cpu={memory_per_cpu}G
#SBATCH --time={run_time}
#SBATCH --array=1-{n_jobs}

# add python
module load palma/2021a
module load Miniconda3

# activate conda env
eval "$(conda shell.bash hook)"
conda activate {conda_env}


python ../project.py --project-folder ../../{self.project_folder} --analysis-name {name} --n-perms {n_perms} --slurm-job-id $SLURM_ARRAY_TASK_ID --n-perms-per-job {n_perms_per_job} --random-state {random_state}{sequential_arguments}
"""
        with open(os.path.join(analysis_folder, "slurm_job.cmd"), "w") as text_file:
            text_file.write(cmd)

        return

    def prepare_staged_slurm_permutation_test(
        self,
        name: str,
        n_perms: int,
        conda_env: str,
        memory_per_cpu: int,
        n_jobs_per_stage: int,
        n_perms_per_stage: int,
        sequential_metric: str,
        run_time: str = "0-01:00:00",
        random_state: int = 1,
        stop_above_p: Optional[float] = None,
        max_exceedances: Optional[int] = None,
    ) -> None:
        """
        Prepare a staged SLURM permutation test that can stop between stages.

        A single large array gains little from sequential stopping: if every
        task starts at once, they all check the budget before any results
        exist and none of them can stop. Splitting the permutations into
        stages fixes that. Each stage is an array job that depends on the
        previous one, so by the time a later stage starts, the earlier results
        are on disk. Its tasks check the budget before doing any work and exit
        within seconds if it has been spent.

        Stages are chained with ``--dependency=afterany`` rather than gated by
        a separate job, which keeps the mechanism simple: later stages are
        still scheduled, they just do nothing. The queue slots are wasted; the
        compute is not.

        Size a stage at roughly the number of permutations a null analysis
        needs — about ``max_exceedances / 0.5`` if the null p-values sit near
        0.5, so ~200 permutations for ``stop_above_p=0.1`` with 1000 planned.

        Parameters
        ----------
        name : str
            Name of the analysis.
        n_perms : int
            Total number of permutation runs across all stages.
        conda_env : str
            Conda environment to activate in the job.
        memory_per_cpu : int
            Memory per CPU in GB.
        n_jobs_per_stage : int
            Number of array tasks in each stage.
        n_perms_per_stage : int
            Number of permutations covered by each stage. Must be divisible by
            `n_jobs_per_stage`.
        sequential_metric : str
            Metric the stopping rule is applied to.
        run_time : str, optional
            Wall time per array task, by default "0-01:00:00".
        random_state : int, optional
            Base random state, by default 1.
        stop_above_p : float, optional
            p-value above which sampling may stop.
        max_exceedances : int, optional
            The same rule expressed as an exceedance count.

        Raises
        ------
        ValueError
            If the analysis folder does not exist, or the stage sizes do not
            divide evenly.
        """
        if name not in os.listdir(self.project_folder):
            raise ValueError(
                f"Analysis {name} not found in project folder {self.project_folder}"
            )

        if n_perms_per_stage % n_jobs_per_stage:
            raise ValueError(
                f"n_perms_per_stage ({n_perms_per_stage}) must be divisible by "
                f"n_jobs_per_stage ({n_jobs_per_stage})."
            )

        budget = self.resolve_exceedance_budget(n_perms, stop_above_p,
                                                max_exceedances)
        n_perms_per_job = n_perms_per_stage // n_jobs_per_stage
        n_stages = int(np.ceil(n_perms / n_perms_per_stage))

        analysis_folder = os.path.join(self.project_folder, name)
        os.makedirs(os.path.join(analysis_folder, "logs"), exist_ok=True)

        shutil.copyfile(
            os.path.abspath(__file__),
            os.path.join(self.project_folder, os.path.basename(__file__)),
        )

        stage_script = f"""#!/bin/bash

#SBATCH --job-name={name}_perm_stage
#SBATCH --output=logs/stage_${{STAGE}}_job_%a.log

#SBATCH --partition normal
#SBATCH --mem-per-cpu={memory_per_cpu}G
#SBATCH --time={run_time}
#SBATCH --array=1-{n_jobs_per_stage}

# add python
module load palma/2021a
module load Miniconda3

# activate conda env
eval "$(conda shell.bash hook)"
conda activate {conda_env}

python ../project.py --project-folder ../../{self.project_folder} \
    --analysis-name {name} --n-perms {n_perms} \
    --slurm-job-id $SLURM_ARRAY_TASK_ID --n-perms-per-job {n_perms_per_job} \
    --random-state {random_state} --stage $STAGE \
    --n-jobs-per-stage {n_jobs_per_stage} \
    --sequential-metric {sequential_metric} --max-exceedances {budget}
"""
        with open(os.path.join(analysis_folder, "slurm_stage.cmd"), "w") as text_file:
            text_file.write(stage_script)

        submit_script = f"""#!/bin/bash
# Submit {n_stages} dependent stages of {n_perms_per_stage} permutations each.
#
# Every stage waits for the previous one to finish, then checks whether the
# exceedance budget for '{sequential_metric}' is already spent. If it is, its
# tasks exit immediately instead of computing anything.

set -euo pipefail

PREVIOUS=""
for STAGE in $(seq 1 {n_stages}); do
    if [ -z "$PREVIOUS" ]; then
        JOB=$(sbatch --parsable --export=ALL,STAGE=$STAGE slurm_stage.cmd)
    else
        JOB=$(sbatch --parsable --dependency=afterany:$PREVIOUS \
                     --export=ALL,STAGE=$STAGE slurm_stage.cmd)
    fi
    echo "stage $STAGE submitted as job $JOB"
    PREVIOUS=$JOB
done
"""
        submit_path = os.path.join(analysis_folder, "submit_stages.sh")
        with open(submit_path, "w") as text_file:
            text_file.write(submit_script)
        os.chmod(submit_path, 0o755)

        return

    def generate_report(self):
        self.reporter.collect_results()
        self.reporter.write_report()


def run_perm_job(
    project_folder: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
        help=(
            "Path to project folder which will contain all data and results "
            "used in the permutation test."
        ),
    ),
    analysis_name: str = typer.Option(
        ..., help="Name of the analysis to run the permutation test for."
    ),
    n_perms: int = typer.Option(
        ..., help="Total number of permutation runs across all jobs."
    ),
    slurm_job_id: int = typer.Option(
        ..., help="SLURM array task ID of the current job."
    ),
    n_perms_per_job: int = typer.Option(
        ..., help="Number of permutation runs to execute in this SLURM job."
    ),
    random_state: int = typer.Option(
        1, help="Base random state for permutation generation."
    ),
    stage: int = typer.Option(
        1, help="1-based stage index for staged permutation runs."
    ),
    n_jobs_per_stage: int = typer.Option(
        0, help="Array size of one stage; 0 for a single unstaged array."
    ),
    sequential_metric: str = typer.Option(
        None,
        help=(
            "Metric for sequential stopping. When set, a job exits without "
            "computing anything if the exceedance budget is already spent."
        ),
    ),
    stop_above_p: float = typer.Option(
        None,
        help=(
            "Stop once it is clear the p-value is at least this large. "
            "Give this or --max-exceedances, not both."
        ),
    ),
    max_exceedances: int = typer.Option(
        None, help="Exceedance budget, i.e. stop_above_p * n_perms."
    ),
):
    """
    Entry point for SLURM-based permutation jobs.

    This function is intended to be called via ``typer.run`` and executes
    a subset of permutation runs for a given analysis.

    Parameters
    ----------
    project_folder : pathlib.Path
        Path to the project folder containing analyses and data.
    analysis_name : str
        Name of the analysis.
    n_perms : int
        Total number of permutation runs across all jobs.
    slurm_job_id : int
        SLURM array task ID (1-based).
    n_perms_per_job : int
        Number of permutations to run in this job.
    random_state : int, optional
        Base random state, by default 1.
    """
    project = PhotonaiProject(str(project_folder))
    project.run_permutation_test_slurm(
        name=analysis_name,
        n_perms=n_perms,
        slurm_job_id=slurm_job_id,
        n_perms_per_job=n_perms_per_job,
        random_state=random_state,
        stage=stage,
        n_jobs_per_stage=n_jobs_per_stage,
        sequential_metric=sequential_metric,
        stop_above_p=stop_above_p,
        max_exceedances=max_exceedances,
    )


if __name__ == "__main__":
    typer.run(run_perm_job)
