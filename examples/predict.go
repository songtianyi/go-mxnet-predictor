package main

import (
	"fmt"
	"math"
	"io/ioutil"
	"encoding/binary"
	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/songtianyi/go-mxnet-predictor"
)

func main() {
	mb, err := ioutil.ReadFile("mean.matrix")
	if err != nil {
		panic(err)
	}
	rgb3c := make([]float32, len(mb)/4)
	for i := 0;i < len(mb)/4;i++ {
		bits := binary.LittleEndian.Uint32(mb[i*4:(i+1)*4])
		rgb3c[i] = math.Float32frombits(bits)
	}
	//fmt.Println(rgb3c[0: 100])

	filePath := "flowertest.jpg"
	img, err := imgio.Open(filePath)
	if err != nil {
		panic(err)
	}

	resized := transform.Resize(img, 299, 299, transform.Linear)
	//fmt.Println(resized)
	_, err = mxnet.CvtImageTo1DArray(resized, rgb3c)
	if err != nil {
		panic(err)
	}
	fmt.Println()
	//fmt.Println(res)

	symbol, err := ioutil.ReadFile("/data/102flowers-symbol.json")
	if err != nil {
		panic(err)
	}
	params, err := ioutil.ReadFile("/data/102flowers-0260.params")
	if err != nil {
		panic(err)
	}

	p, err := mxnet.CreatePredictor(symbol, params, mxnet.Device{mxnet.CPU_DEVICE, 0}, []mxnet.InputNode{{Key: "data", Shape: []uint32{1, 3, 299, 299}}})
	if err != nil {
		panic(err)
	}
	//if err := p.SetInput("data", res); err != nil {
	//	panic(err)
	//}
	fmt.Println(p)
}
