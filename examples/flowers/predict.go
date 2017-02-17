// Copyright 2016 go-mxnet-predictor Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"fmt"
	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/songtianyi/go-mxnet-predictor/mxnet"
	"github.com/songtianyi/go-mxnet-predictor/utils"
	"io/ioutil"
	"sort"
)

func main() {

	// load mean image from file
	nd, err := mxnet.CreateNDListFromFile("/data/mean.bin")
	if err != nil {
		panic(err)
	}
	// free ndarray list operator before exit
	defer nd.Free()

	// get mean image data from C memory
	item, err := nd.Get(0)
	if err != nil {
		panic(err)
	}
	fmt.Println(item.Key, item.Data[0:10], item.Shape, item.Size)

	// load model
	symbol, err := ioutil.ReadFile("/data/102flowers-symbol.json")
	if err != nil {
		panic(err)
	}
	params, err := ioutil.ReadFile("/data/102flowers-0260.params")
	if err != nil {
		panic(err)
	}

	// create predictor
	p, err := mxnet.CreatePredictor(symbol,
		params,
		mxnet.Device{mxnet.CPU_DEVICE, 0},
		[]mxnet.InputNode{{Key: "data", Shape: []uint32{1, 3, 299, 299}}},
	)
	if err != nil {
		panic(err)
	}
	defer p.Free()

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

	// set input
	if err := p.SetInput("data", res); err != nil {
		panic(err)
	}
	// do predict
	if err := p.Forward(); err != nil {
		panic(err)
	}
	// get predict result
	data, err := p.GetOutput(0)
	if err != nil {
		panic(err)
	}
	idxs := make([]int, len(data))
	for i := range data {
		idxs[i] = i
	}
	as := utils.ArgSort{Args: data, Idxs: idxs}
	sort.Sort(as)
	fmt.Println("result:")
	fmt.Println(as.Args)
	fmt.Println(as.Idxs)
}
