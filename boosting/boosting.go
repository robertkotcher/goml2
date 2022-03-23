package boosting

import (
	"robertkotcher.me/ML2022/dataset"
	"robertkotcher.me/ML2022/decision_tree"
	ptr "robertkotcher.me/ML2022/util"
)

// Options effect trees
type BuildOptions struct {
	Evaluator     decision_tree.Evaluator
	LearningRate  float64
	NumIterations int
}

// BoostingModel is a struct that should be generic enough to fit both gradient and ADA boosting models.
// Contains a root node and then subsequent nodes constructed with BuildOptions
type BoostingModel struct {
	Root       *decision_tree.DecisionNode
	Successors *[]decision_tree.DecisionNode
}

// Predict traverses the decisiontrees in this BoostingModel and returns prediction.
func (m BoostingModel) Predict(row dataset.Row) (*float64, error) {
	//
	// TODO: implement me
	//
	return nil, nil
}

// BuildGradiantBoostingModel returns a pointer to BoostingModel. It uses 'evaluator' to determine whether this is boosting or regression.
// The parameter 'options' contains parameters that are specific to the gradient boosting algorithm.
func BuildGradiantBoostingModel(ds *dataset.Dataset, evaluator decision_tree.Evaluator, options BuildOptions) (*BoostingModel, error) {
	model := BoostingModel{}

	rootOptions := decision_tree.BuildOptions{MaxDepth: ptr.PointToInt(1)}
	root, err := decision_tree.BuildTreeWithOverfitting(ds, options.Evaluator, rootOptions)
	if err != nil {
		return nil, err
	}

	// the root is the prediction returned by a single DT, built from 'evaluator', with a depth of 1.
	model.Root = root

	for i := 0; i < options.NumIterations; i++ {
		// 1 clone dataset

		// 2 fit the target column to equal pseudo-residual (model.Predict - row.Y)

		// 3 build a new decision tree on this dataset
		//   on first iteration, root prediction plus this tree's prediction would give us exact target

		// 4 (now this model's prediction would be root prediction plus residual * learning rate)
	}

	return &model, nil
}
