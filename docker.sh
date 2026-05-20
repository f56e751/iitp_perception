#!/bin/bash

docker run -it --network host --gpus all --ipc=host -v $PWD:/mnt --name iitp chaehyeonsong/grounded_sam:latest
