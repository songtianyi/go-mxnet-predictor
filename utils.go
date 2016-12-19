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
	"image"
)

func GetLastError() error {
	if err := C.MXGetLastError(); err != nil {
		return fmt.Errorf(C.GoString(err))
	}
	return nil
}

func CvtImageTo1DArray(src image.Image, meanm []float32) ([]float32, error) {
	if src == nil {
		return nil, fmt.Errorf("src image nil")
	}

	b := src.Bounds()
	h := b.Max.Y - b.Min.Y
	w := b.Max.X - b.Min.X

	if len(meanm) != w*h*3 {
		return nil, fmt.Errorf("mean image matrix invalid")
	}

	res := make([]float32, h*w*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, alpha := src.At(x+b.Min.X, y+b.Min.Y).RGBA()
			//res[y*w+x] = float32(r>>8) - meanm[y*w+x]
			//res[w*h+y*w+x] = float32(g>>8) - meanm[w*h+y*w+x]
			//res[2*w*h+y*w+x] = float32(b>>8) - meanm[2*w*h+y*w+x]
			res[y*w+x] = float32(r) - meanm[y*w+x]
			res[w*h+y*w+x] = float32(g) - meanm[w*h+y*w+x]
			res[2*w*h+y*w+x] = float32(b) - meanm[2*w*h+y*w+x]
		}
	}

	return res, nil
}
