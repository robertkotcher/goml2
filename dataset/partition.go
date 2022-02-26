package dataset

type Partition struct {
	ColumnIndex  int
	ColumnName   string
	Value        float64
	IsContinuous bool
	False        *Dataset
	True         *Dataset
}

func (p Partition) EvaluateRow(r Row) bool {
	if p.IsContinuous {
		return r[p.ColumnIndex] > p.Value
	}
	return r[p.ColumnIndex] == p.Value
}
