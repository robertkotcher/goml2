package decision_tree

import (
	"fmt"
	"math"
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

// GetAlphaIndexFromCrossValidation splits the training data into 10 folds, i.e. into
// 10 groups of train/test subsets. For each training set, we build trees and alpha, and
// determine which tree performs best. In the end, we average the 10 best alphas and determine
// which alpha it's closet to in the incoming 'alphas' slice, returning the index. This index
// can be used to select a tree constructed from _all_ of the training data, pruned with
// cost-complexity pruning.
func GetAlphaIndexFromCrossValidation(alphas []float64, ds *dataset.Dataset, evaluator Evaluator) (*int, error) {
	// (1) build trainsets, testsets - each has 10 items
	trainsets, testsets, err := ds.CrossValidationSets()
	if err != nil {
		return nil, err
	}

	sumalphas := 0.0
	for i, trainset := range trainsets {
		testset := testsets[i]

		// build tree with training set
		tree, _ := BuildTreeWithOverfitting(&trainset, evaluator)
		subtrees, alphas, err := tree.GetSubtreesAndAlphas()
		if err != nil {
			return nil, err
		}

		// evaluate subtrees with test set
		lowesterr := math.MaxFloat64
		lowestalpha := math.MaxFloat64
		for j, st := range *subtrees {
			sterr := 0.0
			for _, testrow := range testset.Rows {
				pred, err := st.Predict(testrow.X())
				if err != nil {
					return nil, err
				}

				sterr += evaluator.GetSingleError(*pred, testrow.Y())
			}
			if sterr < lowesterr {
				lowesterr = sterr
				lowestalpha = (*alphas)[j]
			}
		}
		sumalphas += lowestalpha
	}
	averageAlpha := sumalphas / 10.0

	// find the alpha that we're closest to from passed alphas
	for a, alpha := range alphas {
		if averageAlpha <= alpha {
			// if best alpha is 0 (lowest possible), then we'll hit it at index 0
			return &a, nil
		}
	}

	return nil, fmt.Errorf("average alpha during cross validation was greater than all eligible alphas")
}

// GetSubtreesAndAlphas builds a list of subtrees and the alphas that would be required
// to construct them, given the minimization problem score = R(T) + alpha * (nLeaves)
func (root *DecisionNode) GetSubtreesAndAlphas() (*[]DecisionNode, *[]float64, error) {
	alphas := []float64{0.0}
	subtrees := []DecisionNode{*root.DeepClone()}

	// while the last subtree in subtrees is more than a single node...
	for subtrees[len(subtrees)-1].L != nil && subtrees[len(subtrees)-1].R != nil {
		inputSubtree := subtrees[len(subtrees)-1]

		outSubtree, outAlpha, err := getSubtreesAndAlphas(&inputSubtree, &inputSubtree)
		if err != nil {
			return nil, nil, err
		}

		alphas = append(alphas, *outAlpha)
		subtrees = append(subtrees, *outSubtree)
	}

	return &subtrees, &alphas, nil
}

// getSubtreesAndAlphas returns a *DecisionNode, which is what the tree would look like,
// starting at the original root, if it didn't have the node with the lowest g(t) (or alpha),
// which is the second return value. Basically the returned *DecisionNode is a snapshot
// of the best tree so far, which is overwritten when a new, better, alpha is found.
func getSubtreesAndAlphas(root, iterNode *DecisionNode) (*DecisionNode, *float64, error) {
	// if it's just a leaf, return all nils because there is no subtree or alpha
	// NOTE the smallest tree we can prune AND get a value back is a tree with 3 nodes. In
	// this case we'll get back the parent node
	if iterNode.L == nil && iterNode.R == nil {
		return nil, nil, nil
	}

	iterLeaves, err := iterNode.getLeaves()
	if err != nil {
		return nil, nil, err
	}

	// get the current internal node's error (single-node view, as if it were leaf)
	// --> this is R(t)
	// --> results weighted based on proportion of total data points that made it
	//     to this node.
	iterAsLeafErr, err := iterNode.Evaluator.GetErrorAtNode(iterNode)
	if err != nil {
		return nil, nil, err
	}
	iterWeight := float64(iterNode.TrainData.Size()) / float64(root.TrainData.Size())
	iterAsRootR := *iterAsLeafErr * iterWeight

	// iterate through all leaf nodes under iterNode and calculate the same thing as above,
	// this time adding the results to get a total error within individual errors weighted
	// --> this is R(T_t) = \sum{R(leaves of T_t)}
	// --> results also weighted
	iterAsBranchR := 0.0
	for _, l := range *iterLeaves {
		leafErr, err := iterNode.Evaluator.GetErrorAtNode(&l)
		if err != nil {
			return nil, nil, err
		}
		leafWeight := float64(l.TrainData.Size()) / float64(root.TrainData.Size())
		iterAsBranchR += *leafErr * leafWeight
	}

	// current g value (alpha) - compare with left and right children, and if its better
	bestAlpha := (iterAsRootR - iterAsBranchR) / float64(len(*iterLeaves)-1)

	// get a snapshot of what the subtree would look like, if we were to choose it.
	// we temporarily set the left and right children to nil and then deep clone from
	// the root. Finally, we put left and right children back in place.
	lChild := iterNode.L
	rChild := iterNode.R
	iterNode.L = nil
	iterNode.R = nil
	bestSubtree := root.DeepClone()
	iterNode.L = lChild
	iterNode.R = rChild

	if iterNode.L != nil {
		lBestSubtree, lBestAlpha, err := getSubtreesAndAlphas(root, iterNode.L)
		if err != nil {
			return nil, nil, err
		}
		if lBestAlpha != nil && *lBestAlpha <= bestAlpha {
			bestAlpha = *lBestAlpha
			bestSubtree = lBestSubtree
		}
	}

	if iterNode.R != nil {
		rBestSubtree, rBestAlpha, err := getSubtreesAndAlphas(root, iterNode.R)
		if err != nil {
			return nil, nil, err
		}
		if rBestAlpha != nil && *rBestAlpha <= bestAlpha {
			bestAlpha = *rBestAlpha
			bestSubtree = rBestSubtree
		}
	}

	return bestSubtree, &bestAlpha, nil
}

// Predict returns this node's prediction for this vector of features
func (n *DecisionNode) Predict(row dataset.Row) (*float64, error) {
	expectedNumCols := len(n.TrainData.Rows[0]) - 1
	if len(row) != expectedNumCols {
		return nil, fmt.Errorf("could not predict, expected %d columns, had %d", expectedNumCols, len(row))
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

// getLeaves returns the leaves, starting at n. Note that the leaves
// are _not_ cloned, so they shouldn't get manipulated in any way
func (n *DecisionNode) getLeaves() (*[]DecisionNode, error) {
	if n.L == nil && n.R == nil {
		return &[]DecisionNode{*n}, nil
	}
	leaves := []DecisionNode{}
	if n.L != nil {
		lLeaves, err := n.L.getLeaves()
		if err != nil {
			return nil, err
		}
		leaves = append(leaves, *lLeaves...)
	}
	if n.R != nil {
		rLeaves, err := n.R.getLeaves()
		if err != nil {
			return nil, err
		}
		leaves = append(leaves, *rLeaves...)
	}
	return &leaves, nil
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
	// logrus.Infof("%vtrain data: %v", tabs, n.TrainData)
	logrus.Infof("%vprediction: %v", tabs, n.Evaluator.Predict(n))
	logrus.Infof("%vnum leaf: %d", tabs, n.TrainData.Size())
	logrus.Infof("%simpurity: %f", tabs, n.TrainData.GiniImpurity())
	logrus.Info()

	if n.L != nil {
		n.L.print("L", level+1)
	}
	if n.R != nil {
		n.R.print("R", level+1)
	}
}
