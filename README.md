# PHOTONAI Analysis

- uses Apptainer (https://apptainer.org/docs/user/main/build_a_container.html)
- main.py describes the run script that will be executed when running the container
- main.py uses typer to create a CLI

## Example Usage
```
(photonaiAnalysis) nils@Nilss-MBP photonaiAnalysis % python main.py --help                                                
Usage: main.py [OPTIONS]

  Run PHOTONAI Analysis

Options:
  --folder DIRECTORY              Path to project folder containing features
                                  and target.  [required]
  --task [regression|classification]
                                  Specify type of the prediction task
                                  [required]
  --pipeline [basic|advanced|ensemble]
                                  Specify PHOTONAI pipeline.  [required]
  --help                          Show this message and exit.

```