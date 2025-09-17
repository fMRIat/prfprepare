#!/bin/sh
bids_root_dir=$(dirname "$(pwd)")

mkdir -p $bids_root_dir/singularity_home

unset PYTHONPATH; singularity run \
	-H $bids_root_dir/singularity_home \
	-B $bids_root_dir:/base/data  \
	--cleanenv /ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/lab/prfprepare/prfprepare_7.0.sif \
	--config /base/data/config/prfprepare.json \
	--verbose



