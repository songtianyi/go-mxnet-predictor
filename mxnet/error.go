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
)

// get the last error happeneed.
// go binding for MXGetLastError
func GetLastError() error {
	if err := C.MXGetLastError(); err != nil {
		return fmt.Errorf(C.GoString(err))
	}
	return nil
}
