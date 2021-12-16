#!/bin/bash
# docker build --no-cache --tag scitran/freesurfer-recon-all `pwd`
ME=dlinhardt
GEAR=prfprepare
VERSION=0.0.1
docker build --tag $ME/$GEAR:$VERSION .

