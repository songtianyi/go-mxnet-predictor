package utils

// sort ArgSort.Args in decreasing order
// keep Args' original position in ArgSortIdxs
type ArgSort struct {
	Args []float32
	Idxs []int
}

// Implement sort.Interface Len
func (s ArgSort) Len() int { return len(s.Args) }

// Implement sort.Interface Less
func (s ArgSort) Less(i, j int) bool { return s.Args[i] > s.Args[j] }

// Implment sort.Interface Swap
func (s ArgSort) Swap(i, j int) {
	// swap value
	s.Args[i], s.Args[j] = s.Args[j], s.Args[i]
	// swap index
	s.Idxs[i], s.Idxs[j] = s.Idxs[j], s.Idxs[i]
}
