// Copyright 2016 go-mxnet-predictor Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
