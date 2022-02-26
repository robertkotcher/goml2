package dataset

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

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
	EnumMapper         *EnumMapper
	ColumnNames        []string
	ColumnIsContinuous []bool
	Rows               []Row
}

// columnIndex and columnIsCont are used to help build datsets from CSV files.
// see ColumnsToInclude for more information
// columnName is used to pull the numeric value in enum mapper
type columnIndex int
type columnIsCont bool

// ColumnsToInclude is a data structure that helps us build a dataset from a
// CSV file. By including a columnIndex in the map, this column will be included
// in the dataset. columnIsCont tells us whether or not the data is categorical
type ColumnsToInclude map[columnIndex]columnIsCont

// BuildDatasetFromCSV builds a dataset, where each row looks like:
// [prop0, prop1, ..., propN, target]
// we can extract X and Y using row.X() and row.Y()
func BuildDatasetFromCSV(filepath string, cti ColumnsToInclude, targetC columnIndex) (*Dataset, error) {
	columnIndices := []int{}     // fill with indices so we know how to build rows
	columnContinuous := []bool{} // fill with isContinuous values
	columnNames := []string{}    // fill with names of each column
	targetColumn := -1           // cache target index so we add it last
	targetContinuous := false

	for cIndex, isCont := range cti {
		if cIndex == targetC {
			targetColumn = int(cIndex)
			targetContinuous = bool(isCont)
		} else {
			columnIndices = append(columnIndices, int(cIndex))
			columnContinuous = append(columnContinuous, bool(isCont))
		}
	}

	if targetColumn == -1 {
		return nil, fmt.Errorf("you must include target column in ColumnsToInclude")
	}
	columnIndices = append(columnIndices, targetColumn)
	columnContinuous = append(columnContinuous, targetContinuous)
	// END figuring out index template - we'll use this iteratively to build rows

	// load file
	f, err := os.Open("TitanicTrain.csv")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	data, err := csvReader.ReadAll()
	if err != nil {
		return nil, err
	}

	// file columnNames
	for _, i := range columnIndices {
		columnNames = append(columnNames, data[0][i])
	}

	// we need to fill enum mappings
	e := EnumMapper{}
	for r := 1; r < len(data); r++ {
		for c := 0; c < len(columnIndices); c++ {
			if !columnContinuous[c] {
				e.Insert(columnNames[c], data[r][columnIndices[c]])
			}
		}
	}

	// now that we have the map, iterate through rows again and construct Rows
	rows := []Row{}
	for r := 1; r < len(data); r++ {
		var newRow Row
		for c := 0; c < len(columnIndices); c++ {
			if columnContinuous[c] {
				fl, err := strconv.ParseFloat(data[r][c], 64)
				if err != nil {
					return nil, fmt.Errorf("error parsing row %d col %d", r, c)
				}
				newRow = append(newRow, fl)
			} else {
				newRow = append(newRow, e.LookupNumFromName(columnNames[c], data[r][columnIndices[c]]))
			}
		}
		rows = append(rows, newRow)
	}

	ds := NewDataset(columnNames, columnContinuous, rows, &e)
	return ds, nil
}

func NewDataset(names []string, continuous []bool, rows []Row, e *EnumMapper) *Dataset {
	return &Dataset{
		EnumMapper:         e,
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
	return NewDataset(d.ColumnNames, d.ColumnIsContinuous, []Row{}, d.EnumMapper)
}

// PartitionByName ask the dataset to partition itself based on the provided
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
func (d *Dataset) PartitionByName(column string, on float64) (*Partition, error) {
	for i, c := range d.ColumnNames {
		if column == c {
			p := Partition{
				ColumnIndex:  i,
				ColumnName:   column,
				IsContinuous: d.ColumnIsContinuous[i],
				Value:        on,
				False:        d.cloneColumns(),
				True:         d.cloneColumns(),
			}

			for _, row := range d.Rows {
				if p.EvaluateRow(row) {
					p.True.InsertRow(row)
				} else {
					p.False.InsertRow(row)
				}
			}

			return &p, nil
		}
	}

	return nil, fmt.Errorf("could not find column with name %s", column)
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
