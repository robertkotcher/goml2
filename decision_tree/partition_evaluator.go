package decision_tree

import (
	"fmt"
	"math"

	"robertkotcher.me/ML2022/dataset"
)

type PartitionEvaluator interface {
	// GetMinSamplesForSplit returns the number of data points that stops the algorithm from
	// creating child branches
	GetMinSamplesForSplit() int
	// InitialScore will generally be 0 or a large number. This is the number by which we
	// want to improve during partitioning
	InitialScore() float64
	// BestScore tells us when we can no longer improve a node. If a the best partition score
	// is equal to the best score, we make this a leaf (when no early termination or pruning)
	BestScore() float64
	// Eval takes the initial dataset and a partition, and outputs a score representing how
	// well the partition "un-mixes" the data, it's "purity"
	Eval(dataset *dataset.Dataset, partition *dataset.Partition) (*float64, error)
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

// InitialScore starts out as high as possible, since we're trying to minimize the sum of squared
// residuals
func (r RegressionEvaluator) InitialScore() float64 {
	return math.MaxFloat64
}

// BestScore for regression is 0.0 because there will be no residual between any data point and
// avg target value for each node.
func (r RegressionEvaluator) BestScore() float64 {
	return 0.0
}

// Eval evaluates the current regression split by taking the sum of squared residuals
func (r RegressionEvaluator) Eval(dataset *dataset.Dataset, partition *dataset.Partition) (*float64, error) {
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

// InitialScore is 0 - we want to improve on it with some small, incremental, info gain.
func (c ClassificationEvaluator) InitialScore() float64 {
	return 0.0
}

// BestScore is 1.0 and will happen when node is perfectly pure
func (c ClassificationEvaluator) BestScore() float64 {
	return 1.0
}

// Eval for classification currently only supports information gain metric. An InfoGain of 1
// would mean that this partition perfectly divides the data (bad), and an InfoGain of 0
// would mean we haven't learned anything new from this split (also bad).
func (c ClassificationEvaluator) Eval(dataset *dataset.Dataset, partition *dataset.Partition) (*float64, error) {
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

// IsBetter ...
func (c ClassificationEvaluator) IsBetter(newScore, oldScore float64) bool {
	return newScore > oldScore
}
