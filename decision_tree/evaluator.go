package decision_tree

import (
	"fmt"

	"robertkotcher.me/ML2022/dataset"
)

type Evaluator interface {
	// GetMinSamplesForSplit returns the number of data points that stops the algorithm from
	// creating child branches
	GetMinSamplesForSplit() int
	// Eval takes the initial dataset and a partition, and outputs a score representing how
	// well the partition "un-mixes" the data, it's "purity"
	EvaluateSplit(dataset *dataset.Dataset, partition *dataset.Partition) (*float64, error)
	// GetErrorAtNode is an error score, based on the data the flowed into this node during training
	GetErrorAtNode(root *DecisionNode) (*float64, error)
	// GetSingleError returns the contribution towards a node's error from a single prediction
	GetSingleError(actual, predicted float64) float64
	// Predict return the value that this node predicts
	Predict(node *DecisionNode) float64
	// IsBetter tells us if an evaluation is better than other one (sometimes bigger is better,
	// and sometimes smaller is)
	IsBetter(newScore, oldScore float64) bool
}

// RegressionEvaluator helps us build a regression tree with continuous data
type RegressionEvaluator struct {
	MinSamplesForSplit int
}

// GetMinSamplesForSplit returns the number of data points that stops the algorithm from
// creating child branches
func (r RegressionEvaluator) GetMinSamplesForSplit() int {
	return r.MinSamplesForSplit
}

// EvaluateSplit evaluates the current regression split by taking the sum of squared residuals
func (r RegressionEvaluator) EvaluateSplit(dataset *dataset.Dataset, partition *dataset.Partition) (*float64, error) {
	sumSquareResiduals := 0.0

	falseAvg := 0.0
	for _, row := range partition.False.Rows {
		falseAvg = row[len(row)-1]
	}
	falseAvg = falseAvg / float64(partition.False.Size())
	for _, row := range partition.False.Rows {
		residual := (row[len(row)-1] - falseAvg)
		sumSquareResiduals += (residual * residual)
	}

	trueAvg := 0.0
	for _, row := range partition.True.Rows {
		trueAvg = row[len(row)-1]
	}
	trueAvg = trueAvg / float64(partition.True.Size())
	for _, row := range partition.True.Rows {
		residual := (row[len(row)-1] - trueAvg)
		sumSquareResiduals += (residual * residual)
	}

	return &sumSquareResiduals, nil
}

// GetErrorAtNode evaluates the sum of squared residuals for regression node
func (r RegressionEvaluator) GetErrorAtNode(node *DecisionNode) (*float64, error) {
	totalError := 0.0
	for _, row := range node.TrainData.Rows {
		pred, err := node.Predict(row.X())
		if err != nil {
			return nil, err
		}
		totalError += r.GetSingleError(row.Y(), *pred)
	}
	return &totalError, nil
}

// GetSingleError returns the squared residual
func (r RegressionEvaluator) GetSingleError(actual, predicted float64) float64 {
	return (actual - predicted) * (actual - predicted)
}

// Predict returns the average value for data points at this node
func (r RegressionEvaluator) Predict(node *DecisionNode) float64 {
	total := 0.0
	for _, dp := range node.TrainData.Rows {
		total += dp.Y()
	}
	return total / float64(node.TrainData.Size())
}

func (r RegressionEvaluator) IsBetter(newScore, oldScore float64) bool {
	return newScore < oldScore
}

// ClassificationEvaluator helps us build a decision tree for classification
type ClassificationEvaluator struct {
	MinSamplesForSplit int
}

// GetMinSamplesForSplit returns the number of data points that stops the algorithm from
// creating child branches
func (c ClassificationEvaluator) GetMinSamplesForSplit() int {
	return c.MinSamplesForSplit
}

// EvaluateSplit for classification currently only supports information gain metric. An InfoGain of 1
// would mean that this partition perfectly divides the data (bad), and an InfoGain of 0
// would mean we haven't learned anything new from this split (also bad).
func (c ClassificationEvaluator) EvaluateSplit(dataset *dataset.Dataset, partition *dataset.Partition) (*float64, error) {
	nCols := len(dataset.ColumnIsContinuous)
	if dataset.ColumnIsContinuous[nCols-1] {
		return nil, fmt.Errorf("target labels must not be continuous for classification")
	}

	SImpurity := dataset.GiniImpurity()
	SSize := float64(dataset.Size())

	// left (false) impurity and dataset size
	LImpurity := partition.False.GiniImpurity()
	LSize := float64(partition.False.Size())

	// right (true) impurity and dataset size
	RImpurity := partition.True.GiniImpurity()
	RSize := float64(partition.True.Size())

	avgImpurity := ((RSize / SSize) * RImpurity) + ((LSize / SSize) * LImpurity)
	infoGain := SImpurity - avgImpurity

	return &infoGain, nil
}

// GetErrorAtNode is the total number of misclassified data points at this node divided
// by the total data points that reached this node during training
func (c ClassificationEvaluator) GetErrorAtNode(node *DecisionNode) (*float64, error) {
	nodeClass := node.Evaluator.Predict(node)

	totalError := 0.0
	for _, row := range node.TrainData.Rows {
		if nodeClass != row.Y() {
			totalError += 1
		}
	}
	totalError = totalError / float64(node.TrainData.Size())
	return &totalError, nil
}

// GetSingleError returns 0 if actual equals predicted, and 1 otherwise
func (c ClassificationEvaluator) GetSingleError(actual, predicted float64) float64 {
	if actual == predicted {
		return 0
	}
	return 1
}

// Predict returns the class with the largest representation
func (c ClassificationEvaluator) Predict(node *DecisionNode) float64 {
	counts := map[float64]int{}
	var bestClass float64
	var bestCount int
	for _, r := range node.TrainData.Rows {
		counts[r.Y()] += 1
		if counts[r.Y()] > bestCount {
			bestClass = r.Y()
			bestCount = counts[r.Y()]
		}
	}
	return bestClass
}

// IsBetter ...
func (c ClassificationEvaluator) IsBetter(newScore, oldScore float64) bool {
	return newScore > oldScore
}
