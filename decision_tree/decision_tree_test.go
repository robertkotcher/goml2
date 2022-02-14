package decision_tree

import (
	"testing"

	"robertkotcher.me/ML2022/dataset"
)

func TestDecisionTreeWithClassification(t *testing.T) {
	d := dataset.NewDataset(
		[]string{"color", "diameter", "label"},
		[]bool{false, true, false},
		[]dataset.Row{
			{2.0, 3.0, 3.0},
			{1.0, 3.0, 1.0},
			{2.0, 3.0, 3.0},
			{2.0, 3.0, 3.0},
			{2.0, 3.0, 1.0},
			{3.0, 1.0, 2.0},
			{3.0, 1.0, 2.0},
			{2.0, 3.0, 3.0},
			{2.0, 3.0, 3.0},
		},
	)

	tree, _ := BuildTreeWithOverfitting(d, ClassificationEvaluator{2})

	tree.Print()
}

func TestDecisionTreeWithRegression(t *testing.T) {

	// condition:
	// 0.0 bad
	// 1.0 ok
	// 2.0 great
	d := dataset.NewDataset(
		[]string{"color", "diameter", "label"},
		[]bool{false, true, false},
		[]dataset.Row{
			{2.0, 3.0, 3.0},
			{1.0, 3.0, 1.0},
			{2.0, 3.0, 3.0},
			{2.0, 3.0, 3.0},
			{2.0, 3.0, 1.0},
			{3.0, 1.0, 2.0},
			{3.0, 1.0, 2.0},
			{2.0, 3.0, 3.0},
			{2.0, 3.0, 3.0},
		},
	)

	tree, _ := BuildTreeWithOverfitting(d, RegressionEvaluator{2})

	tree.Print()
}
