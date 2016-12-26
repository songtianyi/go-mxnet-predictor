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

package mxnet

type InputNode struct {
	Key   string   // name
	Shape []uint32 // shape of ndarray
}

const (
	CPU_DEVICE = iota + 1 // cpu device type
	GPU_DEVICE            // gpu device type
)

//TODO higher level api like context
type Device struct {
	Type int // device type
	Id   int // device id
}
