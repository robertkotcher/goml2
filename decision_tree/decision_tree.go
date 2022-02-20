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
	L         *DecisionNode
	R         *DecisionNode
}

// DeepClone ALMOST deep clones this node
//
// NOTE: We do not currently clone the partition. This should not be touched after
// being set.
func (n *DecisionNode) DeepClone() *DecisionNode {
	curr := DecisionNode{Evaluator: n.Evaluator, TrainData: n.TrainData}
	if n.Partition != nil {
		curr.Partition = n.Partition
	}

	if n.L != nil {
		l := n.L.DeepClone()
		curr.L = l
	}
	if n.R != nil {
		r := n.R.DeepClone()
		curr.R = r
	}
	return &curr
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

// GetSubtreeAndMetrics iteratively steps through the entire tree, asking what the SSR
// and num leaves would be if each node was the last one that existed at this branch.
// Returns a list of _all_ possible subtrees, along with their SSRs and numLeaves.
func GetSubtreesAndMetrics(root, iterNode *DecisionNode, evaluator Evaluator) ([]SubtreeAndMetrics, error) {
	var lChild, rChild *DecisionNode
	results := []SubtreeAndMetrics{}

	// pretend that L and R do not exist
	if iterNode.L != nil {
		lChild = iterNode.L
		iterNode.L = nil
	}
	if iterNode.R != nil {
		rChild = iterNode.R
		iterNode.R = nil
	}

	// clone and gather metrics
	ssr, err := evaluator.GetSSR(root)
	if err != nil {
		return nil, err
	}

	results = append(results, SubtreeAndMetrics{
		Root: root.DeepClone(),
		SSR:  ssr,
	})

	// re-attach children and then run recursively on them
	if lChild != nil {
		iterNode.L = lChild
		subResults, err := GetSubtreesAndMetrics(root, lChild, evaluator)
		if err != nil {
			return nil, err
		}
		results = append(results, subResults...)
	}
	if rChild != nil {
		iterNode.R = rChild
		subResults, err := GetSubtreesAndMetrics(root, rChild, evaluator)
		if err != nil {
			return nil, err
		}
		results = append(results, subResults...)
	}

	return results, nil
}

func BuildTreeWithOverfitting(ds *dataset.Dataset, evaluator Evaluator) (*DecisionNode, error) {
	if ds.Size() == 0 {
		return nil, fmt.Errorf("cannot initialize a decision tree node without data")
	}

	outNode := DecisionNode{Evaluator: evaluator, TrainData: ds}

	// check to see if we have too few leaves to split in this node. this is one way to prevent
	// over-fitting
	if ds.Size() < evaluator.GetMinSamplesForSplit() {
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

	// return because no informative partition
	if bestPartition.False.Size() == 0 || bestPartition.True.Size() == 0 {
		return &outNode, nil
	}

	outNode.Partition = bestPartition

	// the Right subtree is built from True partition
	r, err := BuildTreeWithOverfitting(bestPartition.True, evaluator)
	if err != nil {
		return nil, err
	}
	outNode.R = r

	// the Left subtree is built from False partition
	l, err := BuildTreeWithOverfitting(bestPartition.False, evaluator)
	if err != nil {
		return nil, err
	}
	outNode.L = l

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
	logrus.Info()

	if n.L != nil {
		n.L.print("L", level+1)
	}
	if n.R != nil {
		n.R.print("R", level+1)
	}
}

type SubtreeAndMetrics struct {
	SSR       *float64
	NumLeaves int
	Root      *DecisionNode
}
