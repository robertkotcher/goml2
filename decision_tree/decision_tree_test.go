package decision_tree

import (
	"fmt"
	"testing"

	"robertkotcher.me/ML2022/dataset"
)

func TestDecisionTreeWithClassificationSimple(t *testing.T) {
	d := dataset.NewDataset(
		[]string{"in", "echo"},
		[]bool{false, true, false},
		[]dataset.Row{
			{1.0, 1.0},
			{2.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
		},
	)

	classifier := ClassificationEvaluator{3}

	tree, _ := BuildTreeWithOverfitting(d, classifier)
	subtrees, _ := GetSubtreesAndMetrics(tree, tree, classifier)

	for _, s := range subtrees {
		s.Root.print(fmt.Sprintf("Subtree with SSR = %f", *s.SSR), 0)
	}
}

// func TestDecisionTreeWithClassification(t *testing.T) {
// 	d := dataset.NewDataset(
// 		[]string{"color", "diameter", "label"},
// 		[]bool{false, true, false},
// 		[]dataset.Row{
// 			{2.0, 3.0, 3.0},
// 			{1.0, 3.0, 1.0},
// 			{2.0, 3.0, 3.0},
// 			{2.0, 3.0, 3.0},
// 			{2.0, 3.0, 1.0},
// 			{3.0, 1.0, 2.0},
// 			{3.0, 1.0, 2.0},
// 			{2.0, 3.0, 3.0},
// 			{2.0, 3.0, 3.0},
// 		},
// 	)

// 	tree, _ := BuildTreeWithOverfitting(d, ClassificationEvaluator{2}, nil)

// 	tree.Print()
// }

// func TestDecisionTreeWithRegression(t *testing.T) {

// 	// condition:
// 	// 0.0 bad
// 	// 1.0 ok
// 	// 2.0 great
// 	d := dataset.NewDataset(
// 		[]string{"color", "diameter", "label"},
// 		[]bool{false, true, false},
// 		[]dataset.Row{
// 			{2.0, 3.0, 3.0},
// 			{1.0, 3.0, 1.0},
// 			{2.0, 3.0, 3.0},
// 			{2.0, 3.0, 3.0},
// 			{2.0, 3.0, 1.0},
// 			{3.0, 1.0, 2.0},
// 			{3.0, 1.0, 2.0},
// 			{2.0, 3.0, 3.0},
// 			{2.0, 3.0, 3.0},
// 		},
// 	)

// 	tree, _ := BuildTreeWithOverfitting(d, RegressionEvaluator{2}, nil)

// 	tree.Print()
// }
