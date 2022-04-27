#!/bin/bash
# docker build --no-cache

# 1.0.0 first version
# 1.0.1 fixed averaging of different run lengths
#       added option custom_output_name


ME=davidlinhardt
GEAR=prfprepare
VERSION=1.0.1
docker build --tag $ME/$GEAR:$VERSION .

