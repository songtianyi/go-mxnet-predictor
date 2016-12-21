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

type NDList struct {
	handle C.NDListHandle
	size   uint32
}

type NDItem struct {
	Key   string
	Data  []float32
	Shape []uint32
	Ndim  uint32
	Size  uint32
}

func CreateNDListFromFile(filepath string) (*NDList, error) {
	b, err := ioutil.ReadFile(filepath)
	if err != nil {
		return nil, err
	}
	if len(b) < 1 {
		return nil, fmt.Errorf("empty file")
	}
	var (
		handle C.NDListHandle
		size   uint32
	)
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

func (s *NDList) Get(index uint32) (*NDItem, error) {
	var (
		key   *C.char
		data  *C.mx_float
		shape *C.mx_uint
		ndim  C.mx_uint
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
	goshape := (*[1 << 32]uint32)(unsafe.Pointer(shape))[:ndim:ndim]
	for _, v := range goshape {
		size *= v
	}
	godata := (*[1 << 32]float32)(unsafe.Pointer(data))[:size:size]
	return &NDItem{
		C.GoString(key),
		godata,
		goshape,
		uint32(ndim),
		size,
	}, nil
}

func (s *NDList) Free() error {
	success, err := C.MXNDListFree(s.handle)
	if err != nil {
		return err
	} else if success < 0 {
		return GetLastError()
	}
	return nil
}
