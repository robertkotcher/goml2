package decision_tree

import (
	"fmt"
	"strings"

	"github.com/sirupsen/logrus"
	"robertkotcher.me/ML2022/dataset"
)

type DecisionNode struct {
	TrainData *dataset.Dataset
	Partition *dataset.Partition
	L         *DecisionNode
	R         *DecisionNode
}

// BuildTreeWithOverfitting takes a Dataset and PartitionEvaluator and returns a Node that
// has either 0 or 2 children that represent the most informative split, as decided by
// the PartitionEvaluator
func BuildTreeWithOverfitting(ds *dataset.Dataset, evaluator PartitionEvaluator) (*DecisionNode, error) {
	if ds.Size() == 0 {
		return nil, fmt.Errorf("cannot initialize a decision tree node without data")
	}

	outNode := DecisionNode{TrainData: ds}

	// check to see if we have too few leaves to split in this node
	if ds.Size() < evaluator.GetMinSamplesForSplit() {
		return &outNode, nil
	}

	// try to partition this data
	var bestPartition *dataset.Partition
	bestScore := evaluator.InitialScore()
	for c := 0; c < len(ds.ColumnNames)-1; c++ {
		for r := 0; r < len(ds.Rows); r++ {
			name := ds.ColumnNames[c]
			val := ds.Rows[r][c]

			partition := ds.Partition(name, val)

			score, err := evaluator.Eval(ds, &partition)
			if err != nil {
				return nil, err
			}

			if evaluator.IsBetter(*score, bestScore) {
				bestPartition = &partition
				bestScore = *score
			}
		}
	}

	// if best partition results in all things going to a single side, then we
	// know that this is the best division we can make
	if bestPartition == nil || bestPartition.False.Size() == 0 || bestPartition.True.Size() == 0 {
		return &outNode, nil
	}

	outNode.Partition = bestPartition

	l, err := BuildTreeWithOverfitting(bestPartition.False, evaluator)
	if err != nil {
		return nil, err
	}
	outNode.L = l

	r, err := BuildTreeWithOverfitting(bestPartition.True, evaluator)
	if err != nil {
		return nil, err
	}
	outNode.R = r

	return &outNode, nil
}

func (n *DecisionNode) Print() {
	n.print("root", 0)
}

func (n *DecisionNode) print(name string, level int) {
	tabs := strings.Repeat("\t", level)
	logrus.Infof("%vname: %v", tabs, name)
	if n.Partition != nil {
		logrus.Infof("%vsplit: col=%v val=%v", tabs, n.Partition.ColumnName, n.Partition.Value)
	} else {
		logrus.Infof("%vsplit: <nil>", tabs)
	}
	logrus.Infof("%vtrain data: %v", tabs, n.TrainData)
	logrus.Info()

	if n.L != nil {
		n.L.print("L", level+1)
	}
	if n.R != nil {
		n.R.print("R", level+1)
	}
}
