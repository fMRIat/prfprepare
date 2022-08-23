#!/bin/bash
# docker build --no-cache

# 1.0.0 first version
# 1.0.1 fixed averaging of different run lengths
#       added option custom_output_name
# 1.0.2 Removed Freesurfer license writing within the machine because of FS's license and because Singularity does not allow writing within file system
# 1.0.3 added info in the maskinfo.json; bugfixes
# 1.0.4 add option for all subs; fix bug when only doing one session
# 1.0.5 fixed stimulus creation?; options for averaging; do not average tasks with single runs; small fixes

ME=davidlinhardt
GEAR=prfprepare
VERSION=1.0.5
docker build --tag $ME/$GEAR:$VERSION .

