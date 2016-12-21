package mxnet

type InputNode struct {
	Key   string	// name
	Shape []uint32	// shape of ndarray
}

const (
	CPU_DEVICE = iota + 1	// cpu device type
	GPU_DEVICE				// gpu device type
)

//TODO higher level api like context
type Device struct {
	Type int	// device type
	Id   int	// device id
}
