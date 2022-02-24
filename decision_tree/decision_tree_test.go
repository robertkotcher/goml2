package decision_tree

import (
	"testing"

	"github.com/sirupsen/logrus"
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
			{1.0, 1.0},
			{3.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{3.0, 4.0},
			{1.0, 1.0},
			{1.0, 2.0},
			{3.0, 3.0},
			{4.0, 4.0},
			{1.0, 1.0},
			{3.0, 2.0},
			{2.0, 3.0},
			{4.0, 4.0},
			{3.0, 4.0},
		},
	)

	classifier := ClassificationEvaluator{2}

	tree, _ := BuildTreeWithOverfitting(d, classifier)
	trees, alphas, err := tree.GetSubtreesAndAlphas()
	for i := 0; i < len(*alphas); i++ {
		logrus.Infof("--> tree with alpha %f", (*alphas)[i])
		(*trees)[i].Print()
	}
	logrus.Warn(err)

}
