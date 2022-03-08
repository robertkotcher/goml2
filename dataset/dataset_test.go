package dataset

import (
	"testing"
)

func TestCrossValidationSets(t *testing.T) {
	ds := NewDataset(
		[]string{"x", "y"},
		[]bool{true, true},
		[]Row{},
		&EnumMapper{},
	)

	ds.InsertRow(Row{0, 0})
	ds.InsertRow(Row{1, 1})
	ds.InsertRow(Row{2, 2})
	ds.InsertRow(Row{3, 3})
	ds.InsertRow(Row{4, 4})
	ds.InsertRow(Row{5, 5})
	ds.InsertRow(Row{6, 6})
	ds.InsertRow(Row{7, 7})
	ds.InsertRow(Row{8, 8})
	ds.InsertRow(Row{9, 9})

	trainsets, testsets, err := ds.CrossValidationSets()
	if err != nil {
		t.Error(err)
	}

	if trainsets[1].Rows[1][0] != 2 && trainsets[1].Rows[1][1] != 2 {
		t.Error("failed to create the correc train set")
	}

	if testsets[1].Rows[0][0] != 1 && testsets[1].Rows[0][1] != 1 {
		t.Error("failed to create the correc train set")
	}
}
