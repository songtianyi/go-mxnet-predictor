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
	"github.com/songtianyi/go-mxnet-predictor/mxnet"
	"io/ioutil"
)

func main() {

	// load model
	symbol, err := ioutil.ReadFile("./go_test-symbol.json")
	if err != nil {
		panic(err)
	}
	params, err := ioutil.ReadFile("./go_test-0001.params")
	if err != nil {
		panic(err)
	}

	// create predictor
	p, err := mxnet.CreatePredictor(symbol,
		params,
		mxnet.Device{mxnet.CPU_DEVICE, 0},
		[]mxnet.InputNode {
		    mxnet.InputNode { Key: "data", Shape: []uint32{1, 10} },
		    mxnet.InputNode { Key: "softmax_label", Shape: []uint32{1, 10} },
		},
	)
	if err != nil {
		panic(err)
	}

	defer p.Free()
}
