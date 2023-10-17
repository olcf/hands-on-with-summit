#!/bin/bash

#Number of nested conda environments active are stored in CONDA_SHLVL
#Deactivate all conda environments until it reaches 0

if [ -z "${CONDA_SHLVL}" ]
then
    echo "No conda environments ever activated (fresh login)" # CONDA_SHLVL not set
else
    echo "Deactivating conda environments" # CONDA_SHLVL set
    for i in $(seq ${CONDA_SHLVL}); do
        echo ${CONDA_SHLVL}
        conda deactivate
    done
    echo "Done"
fi
