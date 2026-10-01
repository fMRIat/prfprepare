#!/bin/bash
# This script builds the Docker image for the prfprepare tool.

# This script builds the Docker image for the prfprepare tool.

ME=davidlinhardt
GEAR=prfprepare
VERSION=7.7
docker build --platform linux/x86_64 --tag $ME/$GEAR:$VERSION .

