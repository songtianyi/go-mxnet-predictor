## go-mxnet-predictor

[![Build Status](https://travis-ci.org/songtianyi/go-mxnet-predictor.svg?branch=master)](https://travis-ci.org/songtianyi/go-mxnet-predictor)
[![Go Report Card](https://goreportcard.com/badge/github.com/songtianyi/go-mxnet-predictor)](https://goreportcard.com/report/github.com/songtianyi/go-mxnet-predictor)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


go-mxnet-predictor is go binding for mxnet c_predict_api. It's almost as raw as original C api, wish further development for higher level APIs.


## Part 1. Steps to get your own linux dev environment
###### 1.1 Get mxnet and build
	mkdir /root/MXNet/
	cd /root/MXNet/ && git clone https://github.com/dmlc/mxnet.git --recursive
	cd /root/MXNet/mxnet && make -j2
	ln -s $MXNET/mxnet/lib/libmxnet.so /usr/lib/libmxnet.so

###### 1.2 Get go-mxnet-predictor
	go get github.com/anthonynsimon/bild
    go get -u -v github.com/songtianyi/go-mxnet-predictor

A [Dockerfile](https://github.com/songtianyi/docker-dev-envs/blob/master/mxnet.Dockerfile) is offered for building development env

## Part 2. Steps to build flower example
###### 2.1 Get model files, mean.bin and input image and put them in correct path

###### 2.2 Build predict.go
	go build examples/flowers/predict.go

## Part 3. Steps to do inference with go-mxnet-predictor
###### 3.1 Load pre-trained model and create go predictor
	// load model
	symbol, err := ioutil.ReadFile("/data/102flowers-symbol.json")
	if err != nil {
		panic(err)
	}
	params, err := ioutil.ReadFile("/data/102flowers-0260.params")
	if err != nil {
		panic(err)
	}

	p, err := mxnet.CreatePredictor(symbol, params, mxnet.Device{mxnet.CPU_DEVICE, 0}, []mxnet.InputNode{{Key: "data", Shape: []uint32{1, 3, 299, 299}}})
	if err != nil {
		panic(err)
	}
	defer p.Free()
	// see more details in examples/flowers/predict.go

###### 3.2 Load input data and do preprocess
	// load test image for predction
	img, err := imgio.Open("/data/flowertest.jpg")
	if err != nil {
		panic(err)
	}
	// preprocess
	resized := transform.Resize(img, 299, 299, transform.Linear)
	res, err := utils.CvtImageTo1DArray(resized, item.Data)
	if err != nil {
		panic(err)
	}

###### 3.3 Set input data to preditor
	// set input
	if err := p.SetInput("data", res); err != nil {
		panic(err)
	}
###### 3.4 Do prediction
	// do predict
	if err := p.Forward(); err != nil {
		panic(err)
	}

###### 3.5 Get result
	// get predict result
	data, err := p.GetOutput(0)
	if err != nil {
		panic(err)
	}
	// see more details in examples/flowers/predict.go
