package main

import (
	"github.com/sirupsen/logrus"
	"robertkotcher.me/ML2022/dataset"
	"robertkotcher.me/ML2022/decision_tree"
)

func buildTitanicDecisionTree() {
	colData := dataset.ColumnsToInclude{
		"Pclass":   true,
		"Sex":      false,
		"Age":      true,
		"Survived": false,
	}
	ds, err := dataset.BuildDatasetFromCSV("TitanicTrain.csv", colData, "Survived")
	if err != nil {
		logrus.Error(err)
	}

	classifier := decision_tree.ClassificationEvaluator{10}

	tree, _ := decision_tree.BuildTreeWithOverfitting(ds, classifier)
	trees, alphas, err := tree.GetSubtreesAndAlphas()
	if err != nil {
		logrus.Error(err)
	}

	idx, err := decision_tree.GetAlphaIndexFromCrossValidation(*alphas, ds, classifier)
	if err != nil {
		logrus.Error(err)
	}

	logrus.Infof("tree with alpha %f", (*alphas)[len(*trees)-1])
	winner := (*trees)[*idx]
	winner.Print()
}

func buildBostonDecisionTree() {
	colData := dataset.ColumnsToInclude{
		"crim":  true,
		"zn":    true,
		"indus": true,
		"rm":    true,
		"age":   true,
		"tax":   true,
		"medv":  true,
	}

	ds, err := dataset.BuildDatasetFromCSV("BostonHousing.csv", colData, "medv")
	if err != nil {
		logrus.Fatal(err)
	}

	err = ds.VisualizeColumnHistograms("boston_columns.png", 600, 600)
	if err != nil {
		logrus.Fatal(err)
	}

	err = ds.VisualizeColumnVsTarget("boston_scatter.png", 600, 600)
	if err != nil {
		logrus.Fatal(err)
	}

	v, err := ds.VarianceForRows()
	if err != nil {
		logrus.Fatal(err)
	}
	logrus.Info("variance: %v", v)

	classifier := decision_tree.RegressionEvaluator{10}

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

func main() {
	buildBostonDecisionTree()
	// buildTitanicDecisionTree()
}
