#!/bin/sh

export cmd="docker run -ti --rm $4 \
	         -v $2/BIDS/derivatives/fmriprep:/flywheel/v0/input  \
	         -v $2/BIDS/derivatives:/flywheel/v0/output  \
	         -v $2/BIDS:/flywheel/v0/BIDS  \
	         -v $2/$3:/flywheel/v0/config.json \
	         davidlinhardt/prfprepare:$1 "
echo "Launching the following command: "
echo $cmd
eval $cmd

