package decision_tree

import (
	"fmt"
	"strings"

	"github.com/sirupsen/logrus"
	"robertkotcher.me/ML2022/dataset"
)

type DecisionNode struct {
	Evaluator Evaluator
	TrainData *dataset.Dataset
	Partition *dataset.Partition
	SSR       *float64
	L         *DecisionNode
	R         *DecisionNode
}

// Predict returns this node's prediction for each of the rows
func (n *DecisionNode) Predict(row dataset.Row) (*float64, error) {
	expectedNumCols := len(n.TrainData.Rows[0]) - 1
	if len(row) != expectedNumCols {
		return nil, fmt.Errorf("could not predict, expected %d columns", expectedNumCols)
	}

	if n.Partition == nil { // base case 1 - there are no parititions at all (leaf)
		out := n.Evaluator.Predict(n)
		return &out, nil
	}

	var nextChild *DecisionNode
	if n.Partition.EvaluateRow(row) {
		nextChild = n.R
	} else {
		nextChild = n.L
	}

	// base case 2 - could be the case that we're still building the tree and the tree
	// _could_ split the data, but hasn't yet (we're probably at the root). Just run the
	// whole node through the evaluator
	if nextChild == nil {
		out := n.Evaluator.Predict(n)
		return &out, nil
	}

	return nextChild.Predict(row)
}

// BuildTreeWithOverfitting takes a Dataset and PartitionEvaluator and returns a Node that
// has either 0 or 2 children that represent the most informative split, as decided by
// the PartitionEvaluator
//
// root should be passed as nil - it'll be used during recursion so we can calculate the
// current (entire) tree's SSR
func BuildTreeWithOverfitting(ds *dataset.Dataset, evaluator Evaluator, root *DecisionNode) (*DecisionNode, error) {
	if ds.Size() == 0 {
		return nil, fmt.Errorf("cannot initialize a decision tree node without data")
	}

	outNode := DecisionNode{Evaluator: evaluator, TrainData: ds}
	if root == nil {
		root = &outNode
	}

	// check to see if we have too few leaves to split in this node. this is one way to prevent
	// over-fitting
	if ds.Size() < evaluator.GetMinSamplesForSplit() {
		err := evaluator.SetSSR(root, &outNode) // set SSR on pre-prune (early stop)
		if err != nil {
			return nil, err
		}

		return &outNode, nil
	}

	// _always_ partition this data. we might end up with all leaves on one side, which means
	// that this node will be a leaf.
	var bestPartition *dataset.Partition
	var bestScore *float64
	for c := 0; c < len(ds.ColumnNames)-1; c++ {
		for r := 0; r < len(ds.Rows); r++ {
			name := ds.ColumnNames[c]
			val := ds.Rows[r][c]

			partition, err := ds.PartitionByName(name, val)
			if err != nil {
				return nil, err
			}

			score, err := evaluator.EvaluateSplit(ds, partition)
			if err != nil {
				return nil, err
			}

			if bestScore == nil || evaluator.IsBetter(*score, *bestScore) {
				bestPartition = partition
				bestScore = score
			}
		}
	}

	// we only set Partition if there's an informative partition. Partition == nil is a signal
	// that this is a leaf, and that we should evaluate here.
	if bestPartition.False.Size() == 0 || bestPartition.True.Size() == 0 {
		err := evaluator.SetSSR(root, &outNode) // set SSR on leaf
		if err != nil {
			return nil, err
		}

		return &outNode, nil
	}
	outNode.Partition = bestPartition

	// set SSR on internal node _before_ assigning L and R subtrees
	err := evaluator.SetSSR(root, &outNode)
	if err != nil {
		return nil, err
	}

	// the Left subtree is built from False partition
	l, err := BuildTreeWithOverfitting(bestPartition.False, evaluator, root)
	if err != nil {
		return nil, err
	}
	outNode.L = l

	// the Right subtree is built from True partition
	r, err := BuildTreeWithOverfitting(bestPartition.True, evaluator, root)
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
		logrus.Infof("%spartition: col=%v val=%v", tabs, n.Partition.ColumnName, n.Partition.Value)
	} else {
		logrus.Infof("%spartition: <nil>", tabs)
	}
	logrus.Infof("%vtrain data: %v", tabs, n.TrainData)
	logrus.Infof("%vSSR: %f", tabs, *n.SSR)
	logrus.Info()

	if n.L != nil {
		n.L.print("L", level+1)
	}
	if n.R != nil {
		n.R.print("R", level+1)
	}
}
