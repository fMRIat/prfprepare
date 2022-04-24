#!/bin/bash
# docker build --no-cache --tag scitran/freesurfer-recon-all `pwd`
ME=davidlinhardt
GEAR=prfprepare
VERSION=1.0.0
docker build --tag $ME/$GEAR:$VERSION .

