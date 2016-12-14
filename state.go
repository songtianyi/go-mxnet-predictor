package mxnet

/*
// go preamble
#cgo CFLAGS:  -xc++ -std=c++11 -I/root/MXNet/mxnet/include
#cgo LDFLAGS: -L/root/MXNet/mxnet/lib
#include <mxnet/c_api.h>
*/
import "C"

func RandomSeed(int seed) error {
	success, err := C.MXRandomSeed(C.int(seed))
	if err != nil {
		return err
	}else if success < 0 {
		return fmt.Errorf("RandomSeed fail, C.MXRandomSeed return %d", success)
	}
	return nil
}

func NotifyShutdown() error {
	success, err := C.MXNotifyShutdown()
	if err != nil {
		return err
	}else if success < 0 {
		return fmt.Errorf("NotifyShutdown fail, C.MXNotifyShutdown return %d", success)
	}
	return nil
}

