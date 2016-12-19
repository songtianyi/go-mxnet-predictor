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
	"unsafe"
)

type Predictor struct {
	handle C.PredictorHandle
}

func CreatePredictor(symbol []byte,
	params []byte,
	device *Device,
	nodes []InputNode,
) (*Predictor, error) {

	var (
		pc        *C.char
		shapeIdx  = []uint32{0}
		shapeData = []uint32{}
	)

	// malloc a **char which like [][]string to store node keys
	keys := C.malloc(C.size_t(len(nodes)) * C.size_t(unsafe.Sizeof(pc)))
	for i := 0; i < len(nodes); i++ {
		p := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(i)*unsafe.Sizeof(pc)))
		// it will be free later when we free shapeData
		*p = C.CString(nodes[i].Key)

		shapeIdx = append(shapeIdx, uint32(len(nodes[i].Shape)))
		shapeData = append(shapeData, nodes[i].Shape...)
	}

	var handle C.PredictorHandle

	success, err := C.MXPredCreate((*C.char)(unsafe.Pointer(&symbol[0])),
		(*C.char)(unsafe.Pointer(&params[0])),
		C.int(len(params)),
		C.int(device.Type),
		C.int(device.Id),
		C.mx_uint(len(nodes)),
		(**C.char)(keys),
		(*C.mx_uint)(&shapeIdx[0]),
		(*C.mx_uint)(&shapeData[0]),
		handle,
	)

	// free mem we created, go gc won't do that for us
	defer C.free(unsafe.Pointer(keys))
	for i := 0; i < len(nodes); i++ {
		p := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(i)*unsafe.Sizeof(pc)))
		C.free(unsafe.Pointer(*p))
	}
	if err != nil {
		return nil, err
	}
	if success < 0 {
		return nil, fmt.Errorf("Create predictor fail, C.MXPredCreate return %d", success)
	}
	return &Predictor{handle: handle}, nil
}

func (s *Predictor) SetInput(key string, data []float32) error {
	k := C.CString(key)	
	defer C.free(unsafe.Pointer(k))
	if data == nil {
		return fmt.Errorf("intput data nil")
	}
	if n, err := C.MXPredSetInput(s.handle, k, (*C.mx_float)(unsafe.Pointer(&data[0])), C.mx_uint(len(data))); err != nil {
                return err
        } else if n < 0 {
                return GetLastError()
        }
	return nil
}

func (s *Predictor) Forward() error {
	success, err := C.MXPredForward(s.handle)
	if err != nil {
		return err
	} else if success < 0 {
		return fmt.Errorf("Run forward pass fail, C.MXPredForward return %d", success)
	}
	return nil
}

func (s *Predictor) GetOutputShape(index uint32) ([][]uint32, []uint32, error) {
	var (
		shapeData = [][]uint32{{}}
		shapeDim  = []uint32{}
	)
	success, err := C.MXPredGetOutputShape(s.handle,
		C.mx_uint(index),
		(**C.mx_uint)(unsafe.Pointer(&shapeData[0][0])),
		(*C.mx_uint)(unsafe.Pointer(&shapeDim[0])),
	)
	if err != nil {
		return nil, nil, err
	} else if success < 0 {
		return nil, nil, fmt.Errorf("GetOutputShape fail, C.MXPredGetOutputShape return %d", success)
	}
	return shapeData, shapeDim, nil
}

