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

/*
// go preamble
#cgo pkg-config: mxnet
#include <mxnet/c_predict_api.h>
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"io/ioutil"
	"unsafe"
)

// NDArray List operator
type NDList struct {
	handle C.NDListHandle // C handle of NDArray List
	size   uint32
}

// NDArray operator
type NDItem struct {
	Key   string    // name of ndarray
	Data  []float32 // actual data of ndarray
	Shape []uint32  // shape
	Ndim  uint32    // the number of dimension in the shape
	Size  uint32    // Shape[0]*Shape[1]....Shape[Ndim-1]
}

// create NDList from file
// go binding for MXNDListCreate
// MXNDListCreate will load ndarrays from file data
func CreateNDListFromFile(filepath string) (*NDList, error) {
	// read file as binary
	b, err := ioutil.ReadFile(filepath)
	if err != nil {
		return nil, err
	}
	if len(b) < 1 {
		// empty
		return nil, fmt.Errorf("empty file")
	}

	var (
		handle C.NDListHandle // go gc, *handle c gc!
		size   uint32         // go gc
	)
	// create ndarray list from raw bytes
	success, err := C.MXNDListCreate((*C.char)(unsafe.Pointer(&b[0])),
		C.int(len(b)),
		&handle,
		(*C.mx_uint)(unsafe.Pointer(&size)),
	)

	if err != nil {
		return nil, err
	}
	if success < 0 {
		return nil, GetLastError()
	}
	return &NDList{handle: handle, size: size}, nil
}

// create NDList from bytes
// go binding for MXNDListCreate
// MXNDListCreate will load ndarrays from file data
func CreateNDListFromBytes(b []byte) (*NDList, error) {
	if b == nil {
		return nil, fmt.Errorf("input is nil")
	}
	if len(b) < 1 {
		// empty
		return nil, fmt.Errorf("empty input")
	}

	var (
		handle C.NDListHandle // go gc, *handle c gc!
		size   uint32         // go gc
	)
	// create ndarray list from raw bytes
	success, err := C.MXNDListCreate((*C.char)(unsafe.Pointer(&b[0])),
		C.int(len(b)),
		&handle,
		(*C.mx_uint)(unsafe.Pointer(&size)),
	)

	if err != nil {
		return nil, err
	}
	if success < 0 {
		return nil, GetLastError()
	}
	return &NDList{handle: handle, size: size}, nil
}

// get an element from ndarray list
// go binding for MXNDListGet
func (s *NDList) Get(index uint32) (*NDItem, error) {
	var (
		key   *C.char     // pointer to name of the item
		data  *C.mx_float // pointer to ndarray data
		shape *C.mx_uint  // pointer to ndarray shape
		ndim  C.mx_uint   // number of dimension in the shape
	)
	success, err := C.MXNDListGet(s.handle,
		C.mx_uint(index),
		&key,
		&data,
		&shape,
		&ndim,
	)
	if err != nil {
		return nil, err
	} else if success < 0 {
		return nil, GetLastError()
	}

	size := uint32(1)
	// c array to go
	goshape := (*[1 << 32]uint32)(unsafe.Pointer(shape))[:ndim:ndim]
	for _, v := range goshape {
		size *= v
	}
	godata := (*[1 << 32]float32)(unsafe.Pointer(data))[:size:size]
	// NDItem go gc
	return &NDItem{
		C.GoString(key),
		godata,
		goshape,
		uint32(ndim),
		size,
	}, nil
}

// free this NDList's C handle
// go binding for MXNDListFree
func (s *NDList) Free() error {
	success, err := C.MXNDListFree(s.handle)
	if err != nil {
		return err
	} else if success < 0 {
		return GetLastError()
	}
	return nil
}
