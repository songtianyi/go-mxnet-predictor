package utils

type ArgSort struct {
	Args []float32
	Idxs []int
}

func (s ArgSort) Len() int { return len(s.Args) }

func (s ArgSort) Less(i, j int) bool { return s.Args[i] > s.Args[j] }

func (s ArgSort) Swap(i, j int) {
	// exchange value
    s.Args[i], s.Args[j] = s.Args[j], s.Args[i]
	// exchange index
    s.Idxs[i], s.Idxs[j] = s.Idxs[j], s.Idxs[i]
}
