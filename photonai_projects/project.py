import os
import sys
import shutil
import json
import importlib.util
import typer
from pathlib import Path
from typing import Iterable, Literal, Dict, Tuple
from scipy import stats
from itertools import combinations

import numpy as np
import pandas as pd

from photonai.processing import ResultsHandler
from photonai.processing.metrics import Scorer

from photonai_projects.utils import find_latest_photonai_run
from photonai_projects.reporter import Reporter


class PhotonaiProject:
    """
    Manage and compare multiple PHOTONAI analyses within a single project folder.

    This class helps you:
    - create and register new analyses,
    - run PHOTONAI hyperpipes on stored data,
    - run permutation tests (locally or on SLURM),
    - aggregate permutation results,
    - compute permutation-based p-values, and
    - statistically compare multiple analyses (Nadeau–Bengio and permutation-based).
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
        """
        perm_runs = range(n_perms)
        self._run_permutation_test(
            name=name,
            random_state=random_state,
            n_perms=n_perms,
            overwrite=overwrite,
            perm_runs=perm_runs,
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
    # Permutation aggregation / p-values
    # -------------------------------------------------
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
        perm_folder = Path(self.project_folder) / name / "permutations"
        valid_runs, missing_runs = self.check_permutation_test(name, n_perms)

        outer_folds_metrics = []
        for valid_run in valid_runs:
            print(f"Aggregating results for permutation run {valid_run + 1}/{n_perms}")
            handler = ResultsHandler()
            handler.load_from_file(
                str(perm_folder / str(valid_run) / "photonai_results.json")
            )
            mean_metrics = pd.DataFrame(
                handler.get_performance_outer_folds()
            ).mean(axis=0)
            mean_metrics["run"] = valid_run
            outer_folds_metrics.append(mean_metrics)

        perm_results = pd.DataFrame(outer_folds_metrics)

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

        # load data
        X = np.load(os.path.join(data_folder, "X.npy"))
        y = np.load(os.path.join(data_folder, "y.npy"))

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

    def run_permutation_test_slurm(
        self,
        name: str,
        n_perms: int = 1000,
        random_state: int = 15,
        overwrite: bool = False,
        slurm_job_id: int | None = None,
        n_perms_per_job: int | None = None,
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
        """
        perms_to_do = np.arange(
            (slurm_job_id - 1) * n_perms_per_job,
            (slurm_job_id - 1) * n_perms_per_job + n_perms_per_job,
        )
        self._run_permutation_test(
            name=name,
            random_state=random_state,
            n_perms=n_perms,
            overwrite=overwrite,
            perm_runs=perms_to_do,
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


python ../project.py --project-folder ../../{self.project_folder} --analysis-name {name} --n-perms {n_perms} --slurm-job-id $SLURM_ARRAY_TASK_ID --n-perms-per-job {n_perms_per_job} --random-state {random_state}
"""
        with open(os.path.join(analysis_folder, "slurm_job.cmd"), "w") as text_file:
            text_file.write(cmd)

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
    )


if __name__ == "__main__":
    typer.run(run_perm_job)
