#!/bin/sh
base_path=/home_local/dlinhardt/prfprepare_test/data/

docker run -ti --rm $1 \
	-v $base_path/derivatives:/flywheel/v0/input  \
	-v $base_path/derivatives:/flywheel/v0/output  \
	-v $base_path/BIDS:/flywheel/v0/BIDS  \
	-v $base_path/config/prfprepare.json:/flywheel/v0/config.json \
	davidlinhardt/prfprepare:latest
