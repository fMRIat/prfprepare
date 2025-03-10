#!/bin/bash
# docker build --no-cache

# 1.0.0 first version
# 1.0.1 fixed averaging of different run lengths
#       added option custom_output_name
# 1.0.2 Removed Freesurfer license writing within the machine because of FS's license and because Singularity does not allow writing within file system
# 1.0.3 added info in the maskinfo.json; bugfixes
# 1.0.4 add option for all subs; fix bug when only doing one session
# 1.0.5 fixed stimulus creation?; options for averaging; do not average tasks with single runs; small fixes
# 1.0.6 do not average tasks with single runs; fix problem where it could not find the aperture; bugfixes and style improvements
# 1.1.0 style fixes, update to python 3.10, freesurfer to 7.3.2, base docker ubuntu 20.04
# 1.1.1 implemented startScan variable additional to prescanDuration
# 1.2.0 implemented custom annots
# 1.2.1 speedup with large stimulus files, force load in seq as int, fix when no prescan or startScan is defined
# 1.2.2 fix for custom annots
# 1.3.0 first implementation of volume analysis type
# 1.3.1 copy dataset_discription to prfprepare output
# 1.3.2 implemented volume masking based on benson and wang atlas
# 1.3.3 fixed forceParams when multiple tasks are present
# 1.3.4 conda update
# 1.3.5 and again downgrade to python=3.8 and numpy<1.20
# 1.3.6 just skip files that are found in the BIDS folder but not in fmriprep
# 1.3.7 define what is happening if no params.loadMatrix is defined
# 1.4.0 update the python environment
# 1.4.1 debugged the fullBrain option for doing analysis on all voxels/vertices
# 1.5.0 update to ubuntu 24, fixed that is skips bold files without params, significantly reduced image size


ME=davidlinhardt
GEAR=prfprepare
VERSION=1.5.0
docker build --platform linux/x86_64 --tag $ME/$GEAR:$VERSION .

