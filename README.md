## go-mxnet-predictor

[![Build Status](https://travis-ci.org/songtianyi/go-mxnet-predictor.svg?branch=master)](https://travis-ci.org/songtianyi/go-mxnet-predictor)
[![Go Report Card](https://goreportcard.com/badge/github.com/songtianyi/go-mxnet-predictor)](https://goreportcard.com/report/github.com/songtianyi/go-mxnet-predictor)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


go-mxnet-predictor is go binding for mxnet c_predict_api. It's almost as raw as original C api, wish further development for higher level APIs.


## Development envirment
###### Get mxnet and build
	mkdir /root/MXNet/
	cd /root/MXNet/ && git clone https://github.com/dmlc/mxnet.git --recursive
	cd /root/MXNet/mxnet && make -j2
	ln -s $MXNET/mxnet/lib/libmxnet.so /usr/lib/libmxnet.so

###### Get go-mxnet-predictor
	go get github.com/anthonynsimon/bild
    go get -u -v github.com/songtianyi/go-mxnet-predictor

## Steps to build flower example
