import datapane as dp
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


x = np.random.randn(100)
y = np.random.randn(100)
df = pd.DataFrame({'x': x, 'y': y})
plt.figure()
corr_plot = sns.scatterplot(data=df, x="x", y="y")
plt.figure()
hist_plot = sns.histplot(data=df, x='x')

# Text blocks
context = dp.Text(
    """
## Objective
This project has been created as a demo to showcase how Datapane can be used to track, infer and share machine learning pipeline updates. The report has been created as an HTML dashboard which can be easily updated and shared across.
- Age (numeric)
- Sex (text: male, female)
"""
)

report_update = dp.BigNumber(heading="Report last updated", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
accuracy_block = dp.BigNumber(heading="Model accuracy", value="65%", change="2%", is_upward_change=True)

page1 = dp.Page(
    title="Summary",
    blocks=[
        dp.Group(
            context,
            dp.Text("![](https://avatars.githubusercontent.com/u/63720198?s=400&u=17e0e95ba5f7a7a220cfaaadce784094f8429478&v=4)"),
            columns=2,
        ),
        dp.Group(report_update, columns=1),
        dp.Group(accuracy_block, report_update, columns=2),
    ],
)

page2 = dp.Page(
    title="Data profile",
    blocks=[
        dp.Text("""## Dataset details"""),
        dp.Text(
            """This page can be used to display an overview of the dataset along with important elements of the data, features, attributes with null values and so on. """
        ),
        dp.Select(
            blocks=[
                # overview_block,
                dp.Plot(corr_plot, label="Correlation"),
                dp.Plot(hist_plot, label="Histogram"),
                dp.DataTable(df, label="Dataset"),
            ]
        ),
    ],
)

report = dp.View(page1, page2)
dp.save_report(report, path="report.html", open=True)

