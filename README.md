## go-mxnet-predictor

[![Build Status](https://travis-ci.org/songtianyi/go-mxnet-predictor.svg?branch=master)](https://travis-ci.org/songtianyi/go-mxnet-predictor)
[![Go Report Card](https://goreportcard.com/badge/github.com/songtianyi/go-mxnet-predictor)](https://goreportcard.com/report/github.com/songtianyi/go-mxnet-predictor)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


go-mxnet-predictor is go binding for mxnet c_predict_api. It's as raw as original C api, wish further development for higher level APIs. Feel free to join us :)


## Part 1. Steps to build your own linux dev environment
[Dockerfile](https://github.com/songtianyi/docker-dev-envs/blob/master/gmp.Dockerfile) offered for building mxnet and go env. You could skip this part by using Docker

##### 1.1 Install mxnet prerequisites and go
* for mxnet prerequisites check [here](http://mxnet.io/get_started/setup.html#prerequisites)
* for go installation check [here](https://golang.org/doc/install)

##### 1.2 Get mxnet and build
	mkdir /root/MXNet/
	cd /root/MXNet/ && git clone https://github.com/dmlc/mxnet.git --recursive
	cd /root/MXNet/mxnet && make -j2
	ln -s /root/MXNet/mxnet/lib/libmxnet.so /usr/lib/libmxnet.so


## Part 2. Steps to build and run flower example
##### 2.1 Get go-mxnet-predictor and do some configuration
```shell
go get github.com/anthonynsimon/bild
go get -u -v github.com/songtianyi/go-mxnet-predictor
cd $GOPATH/src/github.com/songtianyi/go-mxnet-predictor	
sed -i "/prefix=/c prefix=\/root\/MXNet\/mxnet" travis/mxnet.pc
cp travis/mxnet.pc /usr/lib/pkgconfig/
pkg-config --libs mxnet
```

##### 2.2 Build flowers example
```shell
go build examples/flowers/predict.go
```

##### 2.3 Download example files
To run this example, you need to download model files, mean.bin and input image.
Then put them in correct path. These files are shared in dropbox.

* [102flowers-0260.params](https://www.dropbox.com/s/7l8zye9jpv2bywu/102flowers-0260.params?dl=0)
* [102flowers-symbol.json](https://www.dropbox.com/s/507hikz8561hwxg/102flowers-symbol.json?dl=0)
* [flowertest.jpg](https://www.dropbox.com/s/9ej43gpkcdw3q32/flowertest.jpg?dl=0)
* [mean.bin](https://www.dropbox.com/s/rg45ma97x886i53/mean.bin?dl=0)

##### 2.4 Run example
```shell
./predict
```

## Part 3. Steps to do inference with go-mxnet-predictor
##### 3.1 Load pre-trained model, mean image and create go predictor
```go
// load model
symbol, err := ioutil.ReadFile("/data/102flowers-symbol.json")
if err != nil {
	panic(err)
}
params, err := ioutil.ReadFile("/data/102flowers-0260.params")
if err != nil {
	panic(err)
}

// load mean image from file
nd, err := mxnet.CreateNDListFromFile("/data/mean.bin")
if err != nil {
    panic(err)
}

// free ndarray list operator before exit
defer nd.Free()

// create Predictor
p, err := mxnet.CreatePredictor(symbol, params, mxnet.Device{mxnet.CPU_DEVICE, 0}, []mxnet.InputNode{{Key: "data", Shape: []uint32{1, 3, 299, 299}}})
if err != nil {
	panic(err)
}
defer p.Free()
// see more details in examples/flowers/predict.go
```

##### 3.2 Load input data and do preprocess
```go
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
```

##### 3.3 Set input data to preditor
```go
// set input
if err := p.SetInput("data", res); err != nil {
	panic(err)
}
```
##### 3.4 Do prediction
```go
// do predict
if err := p.Forward(); err != nil {
	panic(err)
}
```

##### 3.5 Get result
```go
// get predict result
data, err := p.GetOutput(0)
if err != nil {
	panic(err)
}
// see more details in examples/flowers/predict.go
```
