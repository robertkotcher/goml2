package dataset

type Row []float64

// X is the train data
func (r Row) X() []float64 {
	out := []float64{}
	for i := 0; i < len(r)-1; i++ {
		out = append(out, r[i])
	}
	return out
}

// Y is the target data
func (r Row) Y() float64 {
	return r[len(r)-1]
}
