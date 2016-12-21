package main

import (
	"fmt"
	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/songtianyi/go-mxnet-predictor/mxnet"
	"github.com/songtianyi/go-mxnet-predictor/utils"
	"io/ioutil"
	"sort"
)

func main() {
	nd, err := mxnet.CreateNDListFromFile("/data/mean.bin")
	if err != nil {
		panic(err)
	}
	defer nd.Free()
	item, err := nd.Get(0)
	if err != nil {
		panic(err)
	}
	fmt.Println(item.Key, item.Data[0:10], item.Shape, item.Size)

	img, err := imgio.Open("/data/flowertest.jpg")
	if err != nil {
		panic(err)
	}
	resized := transform.Resize(img, 299, 299, transform.Linear)
	res, err := utils.CvtImageTo1DArray(resized, item.Data)
	if err != nil {
		panic(err)
	}

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
	defer p.Free()
	if err := p.SetInput("data", res); err != nil {
		panic(err)
	}
	if err := p.Forward(); err != nil {
		panic(err)
	}
	shape, err := p.GetOutputShape(0)
	if err != nil {
		panic(err)
	}
	//fmt.Println(shape)

	size := uint32(1)
	for _, v := range shape {
		size *= v
	}
	data, err := p.GetOutput(0)
	if err != nil {
		panic(err)
	}
	idxs := make([]int, size)
	for i := range data {
		idxs[i] = i
	}
	as := utils.ArgSort{Args: data, Idxs: idxs}
	sort.Sort(as)
	fmt.Println(as)
	fmt.Println("perfect")
}
