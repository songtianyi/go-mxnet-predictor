package main

/*
// go preamble
#cgo CPPFLAGS: -I/data/dockerbuilder/mxnet/mxnet/include/
#cgo LDFLAGS: -L/data/dockerbuilder/mxnet/mxnet/src/
#include <mxnet/c_predict_api.h>
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
		keys      = make([]*C.char, 0)
		shapeIdx  = []C.mx_uint{0}
		shapeData = []C.mx_uint{}
	)
	for i := 0; i < len(nodes); i++ {
		keys = append(keys, C.CString(nodes[i].Key))
		shapeInx = append(shapeInx, C.mx_uint(len(nodes[i].Shape)))
		shapeData = append(shapeData, C.mx_uint(nodes[i].Shape)...)
	}
	var handle C.PredictorHandle

	success, err := C.MXPredCreate((*C.char)(unsafe.Pointer(&symbol[0])),
		(*C.char)(unsafe.Pointer(&params[0])),
		C.int(len(params)),
		C.int(devType),
		C.int(devId),
		C.mx_uint(len(nodes)),
		(**C.char)(keys),
		(*C.mx_uint)(&shapeInx[0]),
		(*C.mx_uint)(&shapeData[0]),
		handle,
	)
	fmt.Println(handle)
}

func main() {
	fmt.Println("fsaf")
}
