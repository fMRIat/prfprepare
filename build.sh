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

ME=davidlinhardt
GEAR=prfprepare
VERSION=1.1.1
docker build --platform linux/x86_64 --tag $ME/$GEAR:$VERSION .

