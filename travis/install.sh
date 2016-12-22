#!/bin/bash
cd travis
git clone https://github.com/dmlc/mxnet.git --recursive
cd mxnet && make -j4
ln -s lib/libmxnet.so /usr/lib/libmxnet.so
go get github.com/anthonynsimon/bild
go get github.com/songtianyi/go-mxnet-predictor
exit 0
