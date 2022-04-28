#!/bin/bash
# docker build --no-cache

# 1.0.0 first version
# 1.0.1 fixed averaging of different run lengths
#       added option custom_output_name
# 1.0.2 Removed Freesurfer license writing within  the machine because of FS's license and because Singularity does not allow writing within file system

ME=davidlinhardt
GEAR=prfprepare
VERSION=1.0.2
docker build --tag $ME/$GEAR:$VERSION .

