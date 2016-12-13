package main
import (
	"fmt"
	"image"
	"os"
	"io"
	"github.com/disintegration/imaging"
)

func main() {
	reader, err := os.Open("test.jpg")
	if err != nil {
		panic(err)
	}
	img, _, _ := image.Decode(reader)
	img = imaging.Fill(img, 224, 224, imagine.Center, imaging.Lanczos)
	symbol, err := ioutil.ReadFile("./Inception-symbol.json")
	if err != nil {
		panic(err)
	}
	params, err := ioutil.ReadFile("./Inception-0009.params")
	if err != nil {
		panic(err)
	}
	synset, err := os.Open("./synset.txt")
	if err != nil {
		panic(err)
	}

	p, err := mxnet.CreatePreditor(symbol, params, mxnet.Device{mxnet.CPU_DEVICE, 0}, []mxnet.InputNode{{Key: "data", Shape: []uint32{1, 3, 224, 224}}})
    if err != nil {
		panic(err)
	}
}
