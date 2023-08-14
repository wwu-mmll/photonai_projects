import os
from datetime import datetime

import numpy as np
import pandas as pd
from glob import glob
import datapane as dp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency

from photonai import Hyperpipe
from photonai.processing import ResultsHandler


class PhotonaiProject:
    def __init__(self, name: str, directory: str = '.'):
        self.name = name
        self.directory = directory
        self.project_dir = os.path.join(self.directory, self.name.replace(" ", "_"))
        self.md_file = os.path.join(self.project_dir, 'README.md')
        self.plot_dir = os.path.join(self.project_dir, 'plots')

        os.makedirs(self.project_dir, exist_ok=True)

        if not os.path.exists(self.md_file):
            with open(self.md_file, 'w') as file:
                content = f"""# PHOTONAI Project {self.name}

Add project description here            
"""
                file.write(content)

    def descriptive_sample_statistics(self, df: pd.DataFrame, title: str, footnote: str,
                                      categorical_target: str, continuous_variables: list,
                                      categorical_variables: list, lancet_format: bool = False):

        list_of_group_tables = list()
        groups = df[categorical_target].unique()
        for group_name in groups:
            df_group = df[df[categorical_target] == group_name]
            table = pd.DataFrame(columns=[group_name])

            # =====================
            # Categories
            # =====================
            for cat_variable in categorical_variables:
                categories = df_group[cat_variable].unique()
                row = pd.DataFrame(index=[cat_variable])
                table = pd.concat([table, row])
                for current_category in categories:
                    n_group = np.sum(df_group[cat_variable] == current_category)
                    n = df_group.shape[0]
                    content = "{} ({:.1%})".format(n_group, n_group / n)
                    row = pd.DataFrame({group_name: content}, index=[current_category])
                    table = pd.concat([table, row])

            # =====================
            # Continuous Variables
            # =====================
            for variable_name in continuous_variables:
                content = "{:.2f} ({:.2f})".format(df_group[variable_name].mean(), df_group[variable_name].std())
                row = pd.DataFrame({group_name: content}, index=[variable_name])
                table = pd.concat([table, row])

            list_of_group_tables.append(table)

        final_table = pd.concat(list_of_group_tables, axis=1)

        # =====================
        # Difference stat test
        # =====================

        diff_test_table = pd.DataFrame(columns=['Difference'])

        for cat_variable in categorical_variables:
            obs = pd.crosstab(index=df[cat_variable], columns=df[categorical_target])[groups]
            chi2, p, dof, ex = chi2_contingency(obs)
            content = self.format_p_value(p)
            row = pd.DataFrame({'Difference': content}, index=[cat_variable])
            diff_test_table = pd.concat([diff_test_table, row])

        for variable_name in continuous_variables:
            p = ttest_ind(a=df.loc[df[categorical_target] == groups[0], variable_name],
                          b=df.loc[df[categorical_target] == groups[1], variable_name], nan_policy='omit').pvalue
            content = self.format_p_value(p)
            row = pd.DataFrame({'Difference': content}, index=[variable_name])
            diff_test_table = pd.concat([diff_test_table, row])

        final_table = pd.concat([final_table, diff_test_table], axis=1)

        # ===========================
        # Handle decimal if necessary
        # ===========================
        if lancet_format:
            final_table = final_table.astype(str)
            for key in final_table.keys():
                final_table[key] = final_table[key].str.replace('.', '·')

        # =====================
        # Title and footnote
        # =====================
        if footnote:
            row = pd.DataFrame(index=[footnote])
            final_table = pd.concat([final_table, row])

        final_table = final_table.T.reset_index().T.reset_index()
        description_row = pd.DataFrame({'index': title}, index=[0])
        final_table = pd.concat([description_row, final_table])

        # last clean-ups
        final_table.iloc[1, 0] = ""
        final_table.iloc[0, 1:] = ""
        final_table.iloc[-1, 1:] = ""

        return final_table

    def format_p_value(self, p):
        if p < 0.001:
            content = "<0.001"
        elif p < 0.01:
            content = "{:.3f}".format(p)
        else:
            content = "{:.2f}".format(p)
        return content

    def save_sample_descriptives(self, df: pd.DataFrame,
                                 categorical_target: str,
                                 continuous_covariates: list,
                                 categorical_covariates: list):
        df.to_csv(os.path.join(self.project_dir, 'df_sample.csv'), index=False)

        table = self.descriptive_sample_statistics(df,
                                              title="Descriptive statistics",
                                              footnote="*t or χ² tests.",
                                              categorical_target=categorical_target,
                                              continuous_variables=continuous_covariates,
                                              categorical_variables=categorical_covariates,
                                              lancet_format=False)
        table.to_csv(os.path.join(self.project_dir, 'sample_description.csv'), index=False)

        # create plots describing the sample
        os.makedirs(self.plot_dir, exist_ok=True)
        plt.figure()
        target_dist = sns.countplot(data=df, x=categorical_target, alpha=0.75)
        plt.savefig(os.path.join(self.plot_dir, 'sample__target_distribution.png'))

        for cov in continuous_covariates:
            plt.figure()
            plot = sns.histplot(data=df, x=cov, hue=categorical_target,
                                kde=True, multiple='dodge')
            plt.savefig(os.path.join(self.plot_dir, f'sample__{cov}_distribution.png'))

        for cov in categorical_covariates:
            plt.figure()
            plot = sns.countplot(data=df, x=categorical_target, hue=cov, alpha=0.75)
            plt.savefig(os.path.join(self.plot_dir, f'sample__{cov}_distribution.png'))

        return table

    def run(self, hyperpipe: Hyperpipe, X: np.ndarray, y: np.ndarray, feature_importances: bool = False):
        hyperpipe.fit(X, y)

    def collect_results(self):
        results = list()
        analysis_folders = glob(os.path.join(self.project_dir, '*/'))
        for analysis in analysis_folders:
            if os.path.basename(os.path.dirname(analysis)) == 'plots':
                continue
            folder = self.find_latest_photonai_run(analysis)

            handler = ResultsHandler()
            handler.load_from_file(os.path.join(folder, "photonai_results.json"))
            df = handler.get_performance_table()
            df.to_csv(os.path.join(folder, "metrics.csv"), index=False)
            df = df.iloc[-1, :]
            best_config_metric = handler.results.hyperpipe_info.best_config_metric
            best_metric = pd.DataFrame({'name': [best_config_metric],
                                       'value': [df[best_config_metric]]})
            best_metric.to_csv(os.path.join(folder, "best_metric.csv"))

            names = df.index.tolist()
            df['analysis'] = os.path.basename(os.path.dirname(analysis))
            df['photonai_folder'] = folder
            df = df[['analysis', 'photonai_folder'] + names]
            results.append(df)
        df = pd.DataFrame(results)
        df = df.drop(columns=['best_config', 'n_train', 'n_validation', 'fold'])
        df.to_csv(os.path.join(self.project_dir, 'summary.csv'), index=False, float_format='%.4f')

    @staticmethod
    def find_latest_photonai_run(folder):
        # find latest calculation of pipeline type
        photonai_runs = glob(os.path.join(folder, "*/"))
        dates = [datetime.strptime(name[-20:-1], '%Y-%m-%d_%H-%M-%S') for name in photonai_runs]
        latest_date = max(dates)

        current_photonai_folder = None
        for tmp_folder in photonai_runs:
            if latest_date.strftime('%Y-%m-%d_%H-%M-%S') in tmp_folder:
                current_photonai_folder = tmp_folder
        return current_photonai_folder

    def add_header(self, header: str, content: list):
        dp_header = dp.Text(f"## {header}")
        blocks = list()
        blocks.append(dp_header)
        blocks.extend(content)
        return dp.Group(blocks=blocks, columns=1)

    def write_report(self):
        df = pd.read_csv(os.path.join(self.project_dir, 'summary.csv'))
        project_page = dp.Group(
            label="Project",
            blocks=[dp.Group(dp.Text(
                    "<img src='https://avatars.githubusercontent.com/u/63720198?s=400&u=17e0e95ba5f7a7a220cfaaadce784094f8429478&v=4' alt='drawing' width='100'/>"),
                    #"![](https://avatars.githubusercontent.com/u/63720198?s=400&u=17e0e95ba5f7a7a220cfaaadce784094f8429478&v=4)"),
                    dp.Text(file=self.md_file),
                    columns=1),
                self.add_header(header="Summary of PHOTONAI Analyses",
                                content=[dp.DataTable(df, label="PHOTONAI Results", caption="Summary of PHOTONAI Analyses")])],
                 columns=1)

        descriptives_page = None
        if os.path.exists(os.path.join(self.project_dir, "df_sample.csv")):
            data_df = pd.read_csv(os.path.join(self.project_dir, 'df_sample.csv'))
            desc_df = pd.read_csv(os.path.join(self.project_dir, 'sample_description.csv'))
            descriptives_table = self.add_header(header="Descriptive Statistics",
                                                 content=[dp.DataTable(desc_df, label="Descriptive Statistics")])
            data_table = self.add_header(header="Sample Data",
                                         content=[dp.DataTable(data_df, label="Sample Data")])
            plot1 = self.add_header(header="Target Distribution", content=[dp.Media(file=os.path.join(self.plot_dir, 'sample__target_distribution.png'),
                             label="Target Distribution")])
            plots = glob(os.path.join(self.plot_dir, '*'))
            dp_plots = list()
            dp_plots.append(plot1)
            for plot in plots:
                if plot == os.path.join(self.plot_dir, 'sample__target_distribution.png'):
                    continue
                else:
                    dp_plots.append(self.add_header(header=f"{os.path.basename(plot).split('sample__')[-1].split('.png')[0].replace('_', ' ')}",
                                                    content=[dp.Media(file=plot,
                             label=f"{os.path.basename(plot)}")]))
            descriptives_page = dp.Group(
                label="Sample Descriptives",
                blocks=[dp.Group(blocks=[data_table, descriptives_table], columns=1),
                    dp.Group(blocks=dp_plots, columns=2)])

        pipelines = list()
        for _, analysis in df.iterrows():
            pipe_results = pd.read_csv(os.path.join(analysis['photonai_folder'], 'metrics.csv'))
            best_metric = pd.read_csv(os.path.join(analysis['photonai_folder'], 'best_metric.csv'))

            best_metric_sem = pipe_results[best_metric['name'][0] + '_sem'].iloc[-1]
            best_metric_group = dp.Group(blocks=[dp.BigNumber(heading=f"{best_metric['name'][0]}", value=f"{best_metric['value'][0]:.2%} [+-{best_metric_sem:.2%}]")],
                                         columns=2)
            pipelines.append(dp.Group(blocks=[
                dp.Text(f"Results for: {analysis['analysis']}"),
                best_metric_group,
                dp.DataTable(pipe_results)], label=f"{analysis['analysis']}"))
        pipe_page = dp.Select(blocks=pipelines, label="PHOTONAI Results", type=dp.SelectType.DROPDOWN)

        report = dp.View(dp.Select(blocks=[project_page, descriptives_page, pipe_page]))
        dp.save_report(report, path=os.path.join(self.project_dir, "report.html"), open=True)


if __name__ == "__main__":
    from data_loader.data_loader import EegResponseData, Target, PreprocessingType, FrequencyBands, RestType
    data_loader = EegResponseData(target=Target.responder)
    df = data_loader.df
    project = PhotonaiProject(name='ML_CPM_pipeline', directory='../../results')
    project.collect_results()
    project.save_sample_descriptives(df=df,
                                     categorical_target=Target.responder,
                                     continuous_covariates=['age', 'BDI'],
                                     categorical_covariates=['sex'])
    project.write_report()