# PHOTONAI Analysis

- uses Apptainer (https://apptainer.org/docs/user/main/build_a_container.html)
- main.py describes the run script that will be executed when running the container
- main.py uses typer to create a CLI

## Example usage of PHOTONAI container
```
apptainer run docker://photonai-analyzer-v2.3.0 --help
apptainer run docker://photonai-analyzer-v2.3.0 --folder ./MYPROJECT/ --task regression --pipeline basic
```

## Example Usage of main.py
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
  --outer-cv INTEGER              Number of outer folds.  [default: 10]
  --inner-cv INTEGER              Number of inner folds.  [default: 10]
  --scaling / --no-scaling        Apply z-scaling of features?  [default:
                                  scaling]
  --imputation / --no-imputation  Apply feature imputation?  [default:
                                  imputation]
  --help                          Show this message and exit.

```