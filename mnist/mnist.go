package mnist

import (
	"fmt"
	"github.com/RenatoGeh/benchmarks/common"
	"github.com/RenatoGeh/benchmarks/params"
	"github.com/RenatoGeh/gospn/data"
	"github.com/RenatoGeh/gospn/learn"
	"github.com/RenatoGeh/gospn/score"
	"github.com/RenatoGeh/gospn/spn"
	"github.com/RenatoGeh/gospn/sys"
)

var (
	rawTest     spn.Dataset
	test        spn.Dataset
	testLabels  []int
	rawTrain    spn.Dataset
	train       spn.Dataset
	trainLabels []int
	scope       map[int]*learn.Variable
	classVar    *learn.Variable
)

func init() {
	fmt.Println("Downloading MNIST-3Bits-2000 dataset...")
	scope, rawTest, rawTrain = data.MNIST3Bits2000()
	_, test, classVar, testLabels = data.ExtractLabels(scope, rawTest)
	_, train, classVar, trainLabels = data.ExtractLabels(scope, rawTrain)
}

func Classify(A params.Algorithm) (*score.S, error) {
	sys.Width, sys.Height = 28, 28
	sys.Max = 8

	score := score.NewScore()
	S := common.ClassStructure(A, rawTrain, scope, trainLabels, classVar)
	score.Evaluate(rawTest, testLabels, S, classVar)
	return score, nil
}

func ClassifyIn(A params.Algorithm, p float64) (*score.S, error) {
	sys.Width, sys.Height = 28, 28
	sys.Max = 8

	allTrain, allLabels := data.Join(rawTrain, rawTest, trainLabels, testLabels)
	D, L := data.PartitionByLabels(allTrain, allLabels, classVar.Categories, []float64{p, 1.0 - p})
	T, R := D[0], D[1]
	tL, rL := L[0], L[1]
	score := score.NewScore()
	S := common.ClassStructure(A, R, scope, rL, classVar)
	score.Evaluate(T, tL, S, classVar)
	return score, nil
}
