package main
import (
	"fmt"
	//"image"
	//"os"
	//"io"
	//"io/ioutil"
	//"github.com/lazywei/go-opencv/opencv"
	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/anthonynsimon/bild/blend"
	//"github.com/songtianyi/go-mxnet-predictor"
)

func main() {
	filePath := "flowertest.jpg"
	//img := opencv.LoadImage(filePath)
	//fmt.Println(img.Channels(), img.Depth(), img.Origin, img.Width(), img.Height())
	//defer img.Release()
	//dst := opencv.CreateImage(
	//	img.Width(),
	//	img.Height(),
	//	img.Depth(), img.Channels())
	//defer dst.Release()
	//opencv.CvtColor(img, dst, 4)
	//resized := opencv.Resize(dst, 299, 299, opencv.CV_INTER_LINEAR)
	//defer resized.Release()

	//m := resized.GetMat()
	//fmt.Println(m.GetData(), m.Rows(), m.Cols(), resized.Width(), resized.Height(), m.Step(), len(m.GetData()))
	
img, err := imgio.Open(filePath)
    if err != nil {
        panic(err)
    }

    resized := transform.Resize(img, 299, 299, transform.Linear)	
	// blend.Subtract()
	

	//symbol, err := ioutil.ReadFile("./Inception-symbol.json")
	//if err != nil {
	//	panic(err)
	//}
	//params, err := ioutil.ReadFile("./Inception-0009.params")
	//if err != nil {
	//	panic(err)
	//}
	//synset, err := os.Open("./synset.txt")
	//if err != nil {
	//	panic(err)
	//}

	//p, err := mxnet.CreatePredictor(symbol, params, &mxnet.Device{mxnet.CPU_DEVICE, 0}, []mxnet.InputNode{{Key: "data", Shape: []uint32{1, 3, 299, 299}}})
    	//if err != nil {
	//	panic(err)
	//}
}
