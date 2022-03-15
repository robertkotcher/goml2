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

	ds.Shuffle()

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

	for i, t := range *trees {
		totres := 0.0
		for _, r := range ds.Rows {
			o, _ := t.Predict(r.X())
			if *o != r.Y() {
				totres += 1.0
			}
		}
		logrus.Infof("alpha %v, num misclassified %v", (*alphas)[i], totres)
	}
}

func buildBostonDecisionTree() {
	colData := dataset.ColumnsToInclude{
		"crim":  true,
		"zn":    true,
		"indus": true,
		"rm":    true,
		"age":   true,
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
	logrus.Infof("variance: %v", v)

	classifier := decision_tree.RegressionEvaluator{10}

	tree, _ := decision_tree.BuildTreeWithOverfitting(ds, classifier)
	trees, alphas, err := tree.GetSubtreesAndAlphas()
	logrus.Warn(err)

	idx, err := decision_tree.GetAlphaIndexFromCrossValidation(*alphas, ds, classifier)
	if err != nil {
		logrus.Error(err)
	}

	logrus.Infof("tree with alpha %f", (*alphas)[len(*trees)-1])
	winner := (*trees)[*idx]
	winner.Print()

	// print key
	logrus.Info("Enum mapper:")
	logrus.Info(*ds.EnumMapper)

	// for i, t := range *trees {
	// 	totres := 0.0
	// 	for _, r := range ds.Rows {
	// 		o, _ := t.Predict(r.X())
	// 		totres += ((*o) - r.Y()) * ((*o) - r.Y())
	// 	}
	// 	totres = totres / float64(ds.Size())
	// 	logrus.Infof("alpha %v, avg residual %v", (*alphas)[i], totres)
	// }

	decision_tree.GetAlphaIndexFromCrossValidation(*alphas, ds, classifier)
}

func main() {
	// buildBostonDecisionTree()
	buildTitanicDecisionTree()
}
