package mxnet

type InputNode struct {
	Key   string
	Shape []uint32
}

const (
	CPU_DEVICE = iota + 1
	GPU_DEVICE
)

type Deivce struct {
	Type int
	Id   int
}
