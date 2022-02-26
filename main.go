package main

import (
	"github.com/sirupsen/logrus"
	"robertkotcher.me/ML2022/dataset"
	"robertkotcher.me/ML2022/decision_tree"
)

func main() {
	colData := dataset.ColumnsToInclude{
		2: true,
		4: false,
		5: true,
		1: false,
	}
	ds, err := dataset.BuildDatasetFromCSV("TitanicTrain.csv", colData, 1)
	if err != nil {
		logrus.Error(err)
	}

	classifier := decision_tree.ClassificationEvaluator{10}

	tree, _ := decision_tree.BuildTreeWithOverfitting(ds, classifier)
	trees, alphas, err := tree.GetSubtreesAndAlphas()
	logrus.Warn(err)

	// print largest alpha
	logrus.Infof("tree with alpha %f", (*alphas)[len(*trees)-1])
	(*trees)[len(*trees)-1].Print()

	// print second largest alpha
	logrus.Infof("tree with alpha %f", (*alphas)[len(*trees)-2])
	(*trees)[len(*trees)-2].Print()

	// print 3rd largest alpha
	logrus.Infof("tree with alpha %f", (*alphas)[len(*trees)-3])
	(*trees)[len(*trees)-3].Print()

	// print key
	logrus.Info("Enum mapper:")
	logrus.Info(*ds.EnumMapper)
}
