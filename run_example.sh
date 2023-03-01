#!/bin/bash

# Run photonaiXperiment using apptainer
apptainer run docker://photonai-v2.3.0 --help
apptainer run docker://photonai-v2.3.0 --folder ./MYPROJECT/ --task regression --pipeline basic
