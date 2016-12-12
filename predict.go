package main

/*
// go preamble
#cgo pkg-config: mxnet
#include <mxnet/c_predict_api.h>
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type InputNode struct {
	Key   string
	Shape []uint32
}

func MXPredCreate(symbol []byte,
	params []byte,
	devType int, devId int,
	nodes []InputNode,
) {

	var (
		pc *C.char
		shapeIdx  = []uint32{0}
		shapeData = []uint32{}
	)

	// malloc for **char which like [][]string to store node keys 
	keys := C.malloc(C.size_t(len(nodes)) * C.size_t(unsafe.Sizeof(pc)))
	for i := 0; i < len(nodes); i++ {
		p := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(i)*unsafe.Sizeof(pc)))
		*p = C.CString(nodes[i].Key)

		shapeIdx = append(shapeIdx, uint32(len(nodes[i].Shape)))
		shapeData = append(shapeData, nodes[i].Shape...)
	}
	var handle C.PredictorHandle

	success, err := C.MXPredCreate((*C.char)(unsafe.Pointer(&symbol[0])),
		(*C.char)(unsafe.Pointer(&params[0])),
		C.int(len(params)),
		C.int(devType),
		C.int(devId),
		C.mx_uint(len(nodes)),
		(**C.char)(keys),
		(*C.mx_uint)(&shapeIdx[0]),
		(*C.mx_uint)(&shapeData[0]),
		handle,
	)

	// free mem we created, go gc won't do that for us
	defer C.free(unsafe.Pointer(keys))
	for i := 0; i < len(nodes);i++ {
		p := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(i)*unsafe.Sizeof(pc)))
		C.free(unsafe.Pointer(*p))
	}
	if err != nil {
		fmt.Println(err)
		return
	}
	if success < 0 {
		return
	}
	fmt.Println(handle)
}

func main() {
	fmt.Println("fsaf")
}
