#!/bin/bash
ls
cd travis
git clone https://github.com/dmlc/mxnet.git --recursive
cd mxnet/make -j4
go get github.com/anthonynsimon/bild
go get github.com/songtianyi/go-mxnet-predictor
