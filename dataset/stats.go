package dataset

import (
	"fmt"
)

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

// VarianceForRows prints out the variance that each row has
func (d *Dataset) VarianceForRows() (map[int]float64, error) {
	if d.Size() == 1 {
		return nil, fmt.Errorf("cannot find variance for dataset of size 1")
	}

	nCols := len(d.Rows[0])

	// this loop finds the averages
	averages := map[int]float64{}
	for r, row := range d.Rows {
		for i := 0; i < nCols; i++ {
			averages[i] += row[i]
			if r == d.Size()-1 {
				averages[i] /= float64(d.Size() - 1)
			}
		}
	}

	results := map[int]float64{}
	for r, row := range d.Rows {
		for i := 0; i < nCols; i++ {
			diff := row[i] - averages[i]
			results[i] += (diff * diff)
			if r == d.Size()-1 {
				// finally, divide by N - 1
				results[i] /= float64(d.Size() - 1)
			}
		}
	}

	return results, nil
}
