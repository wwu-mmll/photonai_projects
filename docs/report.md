# Generate an HTML report for your project

## Example code
You can generate an HTML report that summarizes all of your analyses for a project. This makes it easy to get
a quick overview of all results but you can also access more detailed results for every analysis.

```python
# load an existing project
project = PhotonaiProject(name='breast_cancer', directory='./')

# then call the collect_results and write_report() function
project.collect_results()
project.write_report()
```

## Showcase
Here, you can see how such an HTML report looks like. Simply click through the individual tabs and look at the results.
<iframe src="../static/example_report.html"
        width="100%" height="700px"
        sandbox="allow-scripts allow-same-origin"
        style="border:none;">
</iframe>