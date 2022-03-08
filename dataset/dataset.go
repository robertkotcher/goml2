package dataset

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/sirupsen/logrus"
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
type columnName string

// ColumnsToInclude is a data structure that helps us build a dataset from a
// CSV file. By including a columnIndex in the map, this column will be included
// in the dataset. columnIsCont tells us whether or not the data is categorical
type ColumnsToInclude map[columnName]columnIsCont

// BuildDatasetFromCSV builds a dataset, where each row looks like:
// [prop0, prop1, ..., propN, target]
// we can extract X and Y using row.X() and row.Y()
func BuildDatasetFromCSV(filepath string, cti ColumnsToInclude, targetC columnName) (*Dataset, error) {
	// load file
	f, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	data, err := csvReader.ReadAll()
	if err != nil {
		return nil, err
	}
	colNameToIdx := map[string]int{}
	for c, cName := range data[0] {
		colNameToIdx[cName] = c
	}

	columnIndices := []int{}     // fill with indices so we know how to build rows
	columnContinuous := []bool{} // fill with isContinuous values
	columnNames := []string{}    // fill with names of each column
	targetColumn := -1           // cache target index so we add it last
	targetContinuous := false

	// verify that we didn't pass a non-existant key in cti
	for k := range cti {
		var found bool
		for _, cName := range data[0] {
			if columnName(cName) == k {
				found = true
			}
		}
		if !found {
			return nil, fmt.Errorf("column %s does not exist in dataset", k)
		}
	}

	// fill columnNames and columnIndices
	for n, i := range colNameToIdx {
		nm := columnName(n)
		isCont, ok := cti[columnName(n)]
		if ok {
			if nm == targetC {
				targetColumn = i
				targetContinuous = bool(isCont)
			} else {
				columnNames = append(columnNames, n)
				columnIndices = append(columnIndices, i)
				columnContinuous = append(columnContinuous, bool(isCont))
			}
		}
	}

	if targetColumn == -1 {
		return nil, fmt.Errorf("you must include target column in ColumnsToInclude")
	}
	columnNames = append(columnNames, string(targetC))
	columnIndices = append(columnIndices, targetColumn)
	columnContinuous = append(columnContinuous, targetContinuous)
	// END figuring out index template - we'll use this iteratively to build rows

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
	var numSkipped int
	for row := 1; row < len(data); row++ {
		var newRow Row
		includeRow := true
		for c := 0; c < len(columnIndices); c++ {
			col := columnIndices[c]
			val := data[row][col]
			if columnContinuous[c] {
				fl, err := strconv.ParseFloat(val, 64)
				if err != nil {
					// logrus.Warnf("skipping row %d. error parsing \"%v\" (row %d col %d)", row, val, row, col)
					numSkipped += 1
					includeRow = false
				} else {
					newRow = append(newRow, fl)
				}
			} else {
				newRow = append(newRow, e.LookupNumFromName(columnNames[c], val))
			}
		}
		if includeRow {
			rows = append(rows, newRow)
		}
	}

	logrus.Infof("finished building dataset. skipped %d records", numSkipped)

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

// CrossValidationSets returns pointer to 10 training sets and 10 corresponding test sets
func (d *Dataset) CrossValidationSets() (trainsets *[10]Dataset, testsets *[10]Dataset, err error) {
	trainsets = &[10]Dataset{}
	testsets = &[10]Dataset{}

	nTest := int(d.Size() / 10)

	for s := 0; s < 10; s++ {
		testStartIdx := (s * nTest)        // include start
		testEndIdx := testStartIdx + nTest // exclude end

		trainsets[s] = *d.cloneColumns()
		testsets[s] = *d.cloneColumns()
		for r, row := range d.Rows {
			if r >= testStartIdx && r < testEndIdx { // include in test set
				testsets[s].InsertRow(row)
			} else { // include in train set
				trainsets[s].InsertRow(row)
			}
		}
	}

	return trainsets, testsets, nil
}
