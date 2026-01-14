#!/bin/bash -l


/usr/bin/apptainer run --nv \
    --bind /projects \
    --bind $PWD \
    /sw/user/NGC_containers/pytorch_24.09-py3.sif \
    python3 "$@"