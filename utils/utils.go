package utils

import (
	"fmt"
	"image"
)

// convert go Image to 1-dim array
// in the meantime, subtract mean image
func CvtImageTo1DArray(src image.Image, meanm []float32) ([]float32, error) {

	if src == nil {
		return nil, fmt.Errorf("src image nil")
	}

	if meanm == nil || len(meanm) < 1 {
		return nil, fmt.Errorf("mean image invalid")
	}

	b := src.Bounds()
	h := b.Max.Y - b.Min.Y // image height
	w := b.Max.X - b.Min.X // image width

	res := make([]float32, len(meanm))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := src.At(x+b.Min.X, y+b.Min.Y).RGBA()
			res[y*w+x] = float32(r>>8) - meanm[y*w+x]
			res[w*h+y*w+x] = float32(g>>8) - meanm[w*h+y*w+x]
			res[2*w*h+y*w+x] = float32(b>>8) - meanm[2*w*h+y*w+x]
		}
	}
	return res, nil
}
