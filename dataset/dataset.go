package dataset

type Row []float64

// Dataset describes rows of data, where each index i into the row repesents
// data of the same data type.
//
// Each row of a dataset has N columns, and the last element represents the label
// for that row
//
// Continuous data would be used for integer or float values that do not represent
// an enum.
//
// Enums must be converted to integers before being used.
type Dataset struct {
	ColumnNames        []string
	ColumnIsContinuous []bool
	Rows               []Row
}

type Partition struct {
	ColumnName string
	Value      float64
	False      *Dataset
	True       *Dataset
}

func NewDataset(names []string, continuous []bool, rows []Row) *Dataset {
	return &Dataset{
		ColumnNames:        names,
		ColumnIsContinuous: continuous,
		Rows:               rows,
	}
}

// InsertRow inserts a new row into d
func (d *Dataset) InsertRow(r Row) {
	d.Rows = append(d.Rows, r)
}

// Size returns the number of data points in this dataset
func (d *Dataset) Size() int {
	return len(d.Rows)
}

// cloneColumns creates a new dataset with the same columns and types
func (d *Dataset) cloneColumns() *Dataset {
	return NewDataset(d.ColumnNames, d.ColumnIsContinuous, []Row{})
}

// Partition ask the dataset to partition itself based on the provided
// column and "on" value. Partitioning happens like this:
//
// If the column being partitioned is continuous, each data point D for the
// requested column is true if D > "on"
//
// If the column being partitioned is not continuous, each data point D for
// the requested column is true if D == "on"
//
// The rows are added to either a true or false dataset, depending on whether
// its value for the column evaluates to true or false.
//
// returns:
//
// map[bool] {
// 		false: Dataset <all rows where q.ColumnName evaluated to false>
//		true: Dataset <all rows where q.ColumnName evaluated to false>
// }
func (d *Dataset) Partition(column string, on float64) Partition {
	p := Partition{
		ColumnName: column,
		Value:      on,
		False:      d.cloneColumns(),
		True:       d.cloneColumns(),
	}

	for i, c := range d.ColumnNames {
		// we find the column and
		if column == c {
			for _, row := range d.Rows {
				if d.ColumnIsContinuous[i] {
					if row[i] > on {
						p.True.InsertRow(row)
					} else {
						p.False.InsertRow(row)
					}
				} else {
					if row[i] == on {
						p.True.InsertRow(row)
					} else {
						p.False.InsertRow(row)
					}
				}
			}
			break
		}
	}

	return p
}

// GiniImpurity is a measure of how often a data point in the set would be
// _incorrectly_ labeled if a random label from the set was assigned to
// a data point in the set.
//
// Used by CART (classification and regression tree) algorithms
func (d *Dataset) GiniImpurity() float64 {
	classes := map[float64]float64{}

	for _, r := range d.Rows {
		label := r[len(r)-1]
		classes[label] += 1.0
	}

	impurity := 1.0
	for label := range classes {
		ratio := classes[label] / float64(d.Size())
		impurity -= (ratio * ratio)
	}

	return impurity
}
